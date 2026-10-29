using System.Runtime.InteropServices;

namespace VDBLite;

/// <summary>Owns a handle from the native library and frees it once.</summary>
internal abstract class VdbHandle : SafeHandle
{
    protected VdbHandle() : base(IntPtr.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == IntPtr.Zero;
}

internal sealed class DbHandle : VdbHandle
{
    protected override bool ReleaseHandle()
    {
        Native.vdb_db_free(handle);
        return true;
    }
}

internal sealed class CollectionHandle : VdbHandle
{
    protected override bool ReleaseHandle()
    {
        Native.vdb_collection_free(handle);
        return true;
    }
}

internal sealed class BulkHandle : VdbHandle
{
    protected override bool ReleaseHandle()
    {
        Native.vdb_bulk_free(handle);
        return true;
    }
}

/// <summary>The C API (vdblite.h).</summary>
internal static partial class Native
{
    private const string Lib = "vdblite";

    public const uint OpenReadOnly = 1;
    public const uint OpenExclusive = 2;

    [LibraryImport(Lib)] public static partial IntPtr vdb_version();
    [LibraryImport(Lib)] public static partial IntPtr vdb_last_error();
    [LibraryImport(Lib)] public static partial void vdb_string_free(IntPtr s);

    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int vdb_open(string path, uint flags, ulong cacheBytes, out DbHandle db);
    [LibraryImport(Lib)] public static partial void vdb_close(DbHandle db);
    [LibraryImport(Lib)] public static partial void vdb_db_free(IntPtr db);
    [LibraryImport(Lib)] public static partial int vdb_is_read_only(DbHandle db, [MarshalAs(UnmanagedType.U1)] out bool readOnly);
    [LibraryImport(Lib)] public static partial int vdb_collections(DbHandle db, out IntPtr json);

    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int vdb_create_collection(DbHandle db, string name, string options, out CollectionHandle c);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int vdb_get_collection(DbHandle db, string name, out CollectionHandle c);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int vdb_drop_collection(DbHandle db, string name);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int vdb_write(DbHandle db, string ops, float[] vectors, nuint nFloats, out IntPtr json);
    [LibraryImport(Lib)] public static partial int vdb_optimize(DbHandle db);
    [LibraryImport(Lib)] public static partial int vdb_wait_for_maintenance(DbHandle db);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int vdb_compact_file(string path, [MarshalAs(UnmanagedType.U1)] out bool shrank);

    [LibraryImport(Lib)] public static partial void vdb_collection_free(IntPtr c);
    [LibraryImport(Lib)] public static partial int vdb_collection_info(CollectionHandle c, out IntPtr json);
    [LibraryImport(Lib)] public static partial int vdb_collection_stats(CollectionHandle c, out IntPtr json);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int vdb_embed_inputs(CollectionHandle c, string content, string? format, out IntPtr json);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int vdb_plan_embeddings(CollectionHandle c, string doc, out IntPtr json);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int vdb_search(CollectionHandle c, float[] query, nuint dim, string? options, out IntPtr json);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int vdb_get(CollectionHandle c, string key, out IntPtr json);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int vdb_chunks(CollectionHandle c, string key, out IntPtr json);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int vdb_scan(CollectionHandle c, string? filter, out IntPtr json);
    [LibraryImport(Lib)] public static partial int vdb_count(CollectionHandle c, out ulong count);
    [LibraryImport(Lib)] public static partial int vdb_has_lexical(CollectionHandle c, [MarshalAs(UnmanagedType.U1)] out bool has);
    [LibraryImport(Lib)] public static partial int vdb_enable_lexical(CollectionHandle c, [MarshalAs(UnmanagedType.U1)] out bool built);

    [LibraryImport(Lib)] public static partial ulong vdb_default_checkpoint();
    [LibraryImport(Lib)] public static partial int vdb_bulk_load(CollectionHandle c, ulong checkpointEvery, out BulkHandle b);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int vdb_bulk_add(BulkHandle b, string docs, float[] vectors, nuint nFloats, out ulong written);
    [LibraryImport(Lib)] public static partial int vdb_bulk_checkpoint(BulkHandle b);
    [LibraryImport(Lib)] public static partial int vdb_bulk_finish(BulkHandle b);
    [LibraryImport(Lib)] public static partial int vdb_bulk_finished(BulkHandle b, [MarshalAs(UnmanagedType.U1)] out bool finished);
    [LibraryImport(Lib)] public static partial void vdb_bulk_free(IntPtr b);

    /// <summary>Throws for a failed status, with this thread's error message.</summary>
    public static void Check(int status)
    {
        if (status != 0)
        {
            throw new VdbliteException((ErrorCode)status, Marshal.PtrToStringUTF8(vdb_last_error()) ?? "");
        }
    }

    /// <summary>Takes a string the library returned, and frees it.</summary>
    public static string Take(IntPtr s)
    {
        try
        {
            return Marshal.PtrToStringUTF8(s) ?? "";
        }
        finally
        {
            vdb_string_free(s);
        }
    }

    /// <summary>Vectors back to back, for the C API.</summary>
    public static float[] Flatten(IEnumerable<float[]> vectors)
    {
        var list = vectors as IReadOnlyCollection<float[]> ?? vectors.ToList();
        var flat = new float[list.Sum(v => v.Length)];
        var at = 0;
        foreach (var v in list)
        {
            v.CopyTo(flat, at);
            at += v.Length;
        }
        return flat;
    }
}
