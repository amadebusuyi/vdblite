namespace VDBLite;

/// <summary>
/// A bulk load in progress (see <see cref="Collection.BulkLoad"/>). Always call <see cref="Finish"/>:
/// until then, loaded documents can be read but not found by search.
/// </summary>
public sealed class BulkLoad
{
    /// <summary>Chunks between durable commits, by default.</summary>
    public static readonly ulong DefaultCheckpoint = Native.vdb_default_checkpoint();

    private readonly BulkHandle _h;

    internal BulkLoad(Collection collection, BulkHandle h)
    {
        Collection = collection;
        _h = h;
    }

    /// <summary>The collection being loaded.</summary>
    public Collection Collection { get; }

    /// <summary>Inserts or replaces documents (each with one vector per chunk) in one transaction; returns how many were written.</summary>
    public long Add(IEnumerable<(Doc Doc, IReadOnlyList<float[]> Vectors)> docs)
    {
        var items = docs.ToList();
        var json = items.Select(d =>
        {
            var m = WriteOp.DocFields(d.Doc);
            m["vectors"] = d.Vectors.Count;
            return m;
        });
        var flat = Native.Flatten(items.SelectMany(d => d.Vectors));
        Native.Check(Native.vdb_bulk_add(_h, Json.Write(json), flat, (nuint)flat.Length, out var written));
        return (long)written;
    }

    /// <summary>Makes everything added so far durable (only matters in a file opened exclusively).</summary>
    public void Checkpoint() => Native.Check(Native.vdb_bulk_checkpoint(_h));

    /// <summary>
    /// Builds the index over everything added and makes it searchable. Other writes wait until it's done;
    /// reads and searches don't.
    /// </summary>
    public void Finish() => Native.Check(Native.vdb_bulk_finish(_h));

    /// <summary>Whether <see cref="Finish"/> has been called.</summary>
    public bool IsFinished
    {
        get
        {
            Native.Check(Native.vdb_bulk_finished(_h, out var done));
            return done;
        }
    }
}
