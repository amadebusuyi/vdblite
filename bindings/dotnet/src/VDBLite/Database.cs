using System.Runtime.InteropServices;
using System.Text.Json;

namespace VDBLite;

/// <summary>
/// An open vdblite database file. One process writes a file at a time; other processes can open it
/// read-only alongside and see each commit as it lands. Thread-safe: share one instance. Calls block
/// (the heavy ones use every core); wrap them in <c>Task.Run</c> to keep a UI or event loop free.
/// </summary>
/// <example>
/// <code>
/// using var db = Database.Open("notes.vdb");
/// var docs = db.GetOrCreateCollection("docs", new CollectionOptions(768));
/// await docs.UpsertAsync(new Doc("readme", markdown), embedder);
/// var hits = docs.Search(queryVector, new SearchOptions { K = 5 });
/// </code>
/// </example>
public sealed class Database : IDisposable
{
    internal readonly DbHandle Handle;

    private Database(DbHandle handle) => Handle = handle;

    /// <summary>The version of the native library.</summary>
    public static string Version => Marshal.PtrToStringUTF8(Native.vdb_version()) ?? "";

    /// <summary>Opens (or creates) a file for writing; other processes may read it alongside.</summary>
    /// <param name="cacheBytes">Bytes of file pages to cache; 0 for the default (256 MiB).</param>
    public static Database Open(string path, ulong cacheBytes = 0) => Open(path, 0, cacheBytes);

    /// <summary>Opens for writing with no other process allowed, not even readers; faster small commits.</summary>
    public static Database OpenExclusive(string path, ulong cacheBytes = 0) =>
        Open(path, Native.OpenExclusive, cacheBytes);

    /// <summary>Opens an existing file read-only, following the writing process's commits.</summary>
    public static Database OpenReadOnly(string path, ulong cacheBytes = 0) =>
        Open(path, Native.OpenReadOnly, cacheBytes);

    private static Database Open(string path, uint flags, ulong cacheBytes)
    {
        Native.Check(Native.vdb_open(path, flags, cacheBytes, out var h));
        return new Database(h);
    }

    /// <summary>Rewrites a file nobody has open without its free space; true if it shrank.</summary>
    public static bool CompactFile(string path)
    {
        Native.Check(Native.vdb_compact_file(path, out var shrank));
        return shrank;
    }

    /// <summary>Closes the database; later calls on it and its collections throw <see cref="ErrorCode.Closed"/>.</summary>
    public void Close()
    {
        if (!Handle.IsClosed) Native.vdb_close(Handle);
    }

    /// <summary>Closes the database.</summary>
    public void Dispose()
    {
        Close();
        Handle.Dispose();
    }

    /// <summary>Whether the database was opened read-only.</summary>
    public bool IsReadOnly
    {
        get
        {
            Native.Check(Native.vdb_is_read_only(Handle, out var ro));
            return ro;
        }
    }

    /// <summary>The collection names.</summary>
    public IReadOnlyList<string> Collections()
    {
        Native.Check(Native.vdb_collections(Handle, out var json));
        return Json.Read<List<string>>(json);
    }

    /// <summary>Whether a collection exists.</summary>
    public bool HasCollection(string name) => Collections().Contains(name);

    /// <summary>Creates a collection.</summary>
    public Collection CreateCollection(string name, CollectionOptions options)
    {
        Native.Check(Native.vdb_create_collection(Handle, name, Json.Write(options), out var h));
        return new Collection(this, h, name);
    }

    /// <summary>An existing collection; throws <see cref="ErrorCode.NoSuchCollection"/> if there is none.</summary>
    public Collection Collection(string name)
    {
        Native.Check(Native.vdb_get_collection(Handle, name, out var h));
        return new Collection(this, h, name);
    }

    /// <summary>An existing collection, or a new one with <paramref name="options"/>.</summary>
    public Collection GetOrCreateCollection(string name, CollectionOptions options)
    {
        try
        {
            return Collection(name);
        }
        catch (VdbliteException e) when (e.Code == ErrorCode.NoSuchCollection)
        {
            return CreateCollection(name, options);
        }
    }

    /// <summary>Deletes a collection and everything in it.</summary>
    public void DropCollection(string name) => Native.Check(Native.vdb_drop_collection(Handle, name));

    /// <summary>
    /// Applies upserts and deletes atomically, in one transaction:
    /// <c>db.Write(WriteOp.Upsert(docs, doc, vectors), WriteOp.Delete(docs, "old"))</c>.
    /// </summary>
    public IReadOnlyList<WriteResult> Write(params WriteOp[] ops)
    {
        var flat = Native.Flatten(ops.SelectMany(o => o.Vectors));
        Native.Check(Native.vdb_write(Handle, Json.Write(ops.Select(o => o.Json)), flat, (nuint)flat.Length, out var json));
        var res = Json.Read<WriteResponse>(json);
        if (res.Warning is not null)
        {
            // Committed; only sealing the write buffer afterwards failed.
            throw new VdbliteException(ErrorCode.Maintenance, res.Warning);
        }
        return res.Results;
    }

    private sealed record WriteResponse(List<WriteResult> Results, string? Warning);

    /// <summary>Seals buffered chunks and merges small segments in every collection, using every core.</summary>
    public void Optimize() => Native.Check(Native.vdb_optimize(Handle));

    /// <summary>Waits for background sealing and compaction to catch up with every commit so far.</summary>
    public void WaitForMaintenance() => Native.Check(Native.vdb_wait_for_maintenance(Handle));
}

/// <summary>One change for <see cref="Database.Write"/>: an upsert or a delete.</summary>
public sealed class WriteOp
{
    internal readonly Dictionary<string, object?> Json;
    internal readonly IReadOnlyList<float[]> Vectors;

    private WriteOp(Dictionary<string, object?> json, IReadOnlyList<float[]> vectors)
    {
        Json = json;
        Vectors = vectors;
    }

    /// <summary>Inserts or replaces a document with one vector per chunk (see <see cref="Collection.EmbedInputs"/>).</summary>
    public static WriteOp Upsert(Collection collection, Doc doc, IReadOnlyList<float[]> vectors)
    {
        var m = DocJson(collection, doc);
        m["vectors"] = vectors.Count;
        return new WriteOp(m, vectors);
    }

    internal static WriteOp UpsertPlanned(Collection collection, Doc doc, IReadOnlyList<string> hashes, IReadOnlyList<float[]> vectors)
    {
        var m = DocJson(collection, doc);
        m["embeddings"] = hashes;
        return new WriteOp(m, vectors);
    }

    /// <summary>Deletes a document.</summary>
    public static WriteOp Delete(Collection collection, string key) =>
        new(new() { ["op"] = "delete", ["collection"] = collection.Name, ["key"] = key }, []);

    private static Dictionary<string, object?> DocJson(Collection collection, Doc doc)
    {
        var m = DocFields(doc);
        m["op"] = "upsert";
        m["collection"] = collection.Name;
        return m;
    }

    internal static Dictionary<string, object?> DocFields(Doc doc)
    {
        var m = new Dictionary<string, object?> { ["key"] = doc.Key, ["content"] = doc.Content };
        if (doc.Meta is not null) m["meta"] = doc.Meta;
        if (doc.Format is not null) m["format"] = doc.Format;
        return m;
    }
}
