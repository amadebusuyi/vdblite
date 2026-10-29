using System.Text.Json;
using System.Text.Json.Serialization;

namespace VDBLite;

/// <summary>The kind of a <see cref="VdbliteException"/>.</summary>
public enum ErrorCode
{
    /// <summary>The file or a transaction failed, or stored bytes were invalid.</summary>
    Storage = 1,
    /// <summary>A vector had the wrong number of dimensions.</summary>
    Dimension = 2,
    /// <summary>A collection with this name already exists.</summary>
    CollectionExists = 3,
    /// <summary>No collection with this name exists.</summary>
    NoSuchCollection = 4,
    /// <summary>A value passed in was invalid.</summary>
    InvalidArgument = 5,
    /// <summary>The embedder failed.</summary>
    Embedder = 6,
    /// <summary>The document changed after its embeddings were planned.</summary>
    StaleEmbeddings = 7,
    /// <summary>The file's format version isn't supported by this build.</summary>
    Version = 8,
    /// <summary>Sealing or compacting the index failed; nothing committed was lost.</summary>
    Maintenance = 9,
    /// <summary>The database was opened read-only.</summary>
    ReadOnly = 10,
    /// <summary>The database is closed.</summary>
    Closed = 11,
    /// <summary>A bug in vdblite: please report it.</summary>
    Panic = 12,
}

/// <summary>A failed vdblite call; <see cref="Code"/> says what kind.</summary>
public sealed class VdbliteException(ErrorCode code, string message) : Exception($"{code}: {message}")
{
    /// <summary>What kind of failure this is.</summary>
    public ErrorCode Code { get; } = code;
}

/// <summary>Turns texts into vectors, one per text, in order: any embedding model.</summary>
public delegate Task<IReadOnlyList<float[]>> Embedder(IReadOnlyList<string> texts);

/// <summary>A document to write.</summary>
/// <param name="Key">Unique within the collection.</param>
/// <param name="Content">The text: Markdown, HTML, code or plain text.</param>
/// <param name="Meta">Strings, numbers, booleans, null, or lists of those.</param>
/// <param name="Format">"markdown", "code", "html", "text", a file name, extension or MIME type. Sniffed if null.</param>
public sealed record Doc(
    string Key,
    string Content,
    IReadOnlyDictionary<string, object?>? Meta = null,
    string? Format = null);

/// <summary>Options for a new collection. Only <see cref="Dim"/> is required.</summary>
public sealed record CollectionOptions(int Dim)
{
    /// <summary>"cosine" (default), "dot" or "l2".</summary>
    public string? Metric { get; init; }

    /// <summary>"int8" (default: int8 graph walk, f16 re-ranking), "binary" (1-bit graph walk, f16 re-ranking) or "none" (f32).</summary>
    public string? Quantization { get; init; }

    /// <summary>How documents are split; null for structural chunking with default sizes.</summary>
    public Chunking? Chunking { get; init; }

    /// <summary>Chunks buffered before they're sealed into a graph (default 65536).</summary>
    public int? SealThreshold { get; init; }

    /// <summary>Keep a BM25 keyword index for hybrid search (default true).</summary>
    public bool? Lexical { get; init; }

    /// <summary>HNSW: links per node.</summary>
    public int? M { get; init; }

    /// <summary>HNSW: beam width while building.</summary>
    public int? EfConstruction { get; init; }

    /// <summary>HNSW: default beam width while searching.</summary>
    public int? EfSearch { get; init; }
}

/// <summary>How documents are split into chunks.</summary>
/// <param name="Kind">"structural" (along headings, paragraphs, code blocks) or "none" (one chunk per document).</param>
public sealed record Chunking(string Kind = "structural", int? TargetTokens = null, int? OverlapTokens = null)
{
    /// <summary>Each document is one chunk with one vector.</summary>
    public static Chunking None { get; } = new("none");
}

/// <summary>Refines a search.</summary>
public sealed record SearchOptions
{
    /// <summary>Results to return (default 10).</summary>
    public int? K { get; init; }

    /// <summary>Graph beam width: higher is slower and more accurate.</summary>
    public int? Ef { get; init; }

    /// <summary>
    /// A Mongo-style metadata filter, e.g. <c>new() { ["lang"] = "en", ["year"] = new Dictionary&lt;string, object?&gt; { ["$gte"] = 2020 } }</c>.
    /// Operators: $eq $ne $in $nin $gt $gte $lt $lte $exists $and $or $not.
    /// </summary>
    public IReadOnlyDictionary<string, object?>? Filter { get; init; }

    /// <summary>Query text for hybrid search: chunks are also ranked by BM25 keyword match and the rankings fused.</summary>
    public string? Text { get; init; }
}

/// <summary>A chunk that matched a search.</summary>
/// <param name="ByteStart">UTF-8 byte offset of the chunk in the document.</param>
/// <param name="Score">Higher is better: cosine similarity, dot product or negative squared L2; in a hybrid search, the fused rank score.</param>
/// <param name="VectorScore">Null if only the keyword search found this chunk.</param>
/// <param name="LexicalScore">Null if the keyword search didn't match it.</param>
public sealed record ChunkHit(
    string Key, ulong DocId, uint ChunkId, uint Ordinal, int ByteStart, int ByteEnd, float Score,
    float? VectorScore, float? LexicalScore, string Text, IReadOnlyList<string> HeadingPath,
    IReadOnlyDictionary<string, object?> Meta);

/// <summary>A document ranked by its chunks; <paramref name="Chunks"/> are its matching chunks, best first.</summary>
public sealed record DocHit(
    string Key, ulong DocId, float Score, IReadOnlyDictionary<string, object?> Meta, IReadOnlyList<ChunkHit> Chunks);

/// <summary>A hit widened to neighbouring chunks or its section; <paramref name="Ordinals"/> are the chunks covered, [start, end).</summary>
public sealed record Passage(
    string Key, ulong DocId, int ByteStart, int ByteEnd, float Score, string Text, uint[] Ordinals,
    IReadOnlyList<string> HeadingPath, IReadOnlyDictionary<string, object?> Meta);

/// <summary>A stored document with its content.</summary>
public sealed record Document(
    string Key, ulong DocId, string Content, IReadOnlyDictionary<string, object?> Meta, ulong Version,
    ulong CreatedMs, ulong UpdatedMs, string Format, int ChunkCount);

/// <summary>A summary of a stored document.</summary>
public sealed record DocInfo(
    string Key, ulong DocId, IReadOnlyDictionary<string, object?> Meta, ulong Version, ulong CreatedMs,
    ulong UpdatedMs, long ContentLength, int ChunkCount);

/// <summary>One chunk of a stored document.</summary>
public sealed record ChunkInfo(
    uint ChunkId, uint Ordinal, int ByteStart, int ByteEnd, int OverlapStart, IReadOnlyList<string> HeadingPath, int Tokens);

/// <summary>
/// The outcome of one write. For an upsert, <paramref name="Embedded"/> counts the chunks that took new
/// vectors and <paramref name="Changed"/> is false when nothing changed; for a delete, <paramref name="Deleted"/>
/// says whether the document existed.
/// </summary>
public sealed record WriteResult(
    string Op, string Collection, string Key, ulong DocId, ulong Version, int Chunks, int Embedded, bool Changed,
    bool Deleted);

/// <summary>A collection's settings.</summary>
public sealed record CollectionInfo(
    string Name, int Dim, string Metric, string Quantization, string Chunking, int? TargetTokens, int? OverlapTokens);

/// <summary>A collection's index; <paramref name="MaintenanceError"/> is the last background maintenance failure.</summary>
public sealed record IndexStats(long Chunks, int SealedSegments, long Buffered, long MemoryBytes, string? MaintenanceError);

/// <summary>The JSON the native library speaks.</summary>
internal static class Json
{
    public static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Converters = { new PlainObjectConverter() },
    };

    public static string Write<T>(T value) => JsonSerializer.Serialize(value, Options);

    public static T Read<T>(IntPtr s) => JsonSerializer.Deserialize<T>(Native.Take(s), Options)!;

    /// <summary>Reads metadata values as plain .NET values: string, long, double, bool, null, or lists.</summary>
    private sealed class PlainObjectConverter : JsonConverter<object>
    {
        public override object? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) =>
            reader.TokenType switch
            {
                JsonTokenType.String => reader.GetString(),
                JsonTokenType.True => true,
                JsonTokenType.False => false,
                JsonTokenType.Null => null,
                // Cast first: a long : double conditional would make every number a double.
                JsonTokenType.Number => reader.TryGetInt64(out var l) ? (object)l : reader.GetDouble(),
                JsonTokenType.StartArray => JsonSerializer.Deserialize<List<object?>>(ref reader, options),
                _ => JsonSerializer.Deserialize<Dictionary<string, object?>>(ref reader, options),
            };

        public override void Write(Utf8JsonWriter writer, object value, JsonSerializerOptions options) =>
            JsonSerializer.Serialize(writer, value, value.GetType(), options);
    }
}
