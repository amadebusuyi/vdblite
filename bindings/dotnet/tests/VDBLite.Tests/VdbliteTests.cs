using VDBLite;
using Xunit;

public sealed class VdbliteTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("vdblite-test").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    /// <summary>A toy embedder: [length, count of "rust", 1].</summary>
    private static IReadOnlyList<float[]> Embed(IReadOnlyList<string> texts) =>
        texts.Select(t => new float[] { t.Length, t.Split("rust").Length - 1, 1 }).ToList();

    private static Task<IReadOnlyList<float[]>> EmbedAsync(IReadOnlyList<string> texts) => Task.FromResult(Embed(texts));

    private const string Readme = "# Intro\n\nrust rust rust\n\n## Usage\n\nOther words here.";

    [Fact]
    public async Task RoundTrip()
    {
        var path = Path.Combine(_dir, "test.vdb");
        using var db = Database.Open(path);
        Assert.NotEmpty(Database.Version);
        var docs = db.CreateCollection("docs", new CollectionOptions(3));
        Assert.Equal(ErrorCode.CollectionExists,
            Assert.Throws<VdbliteException>(() => db.CreateCollection("docs", new CollectionOptions(3))).Code);

        var meta = new Dictionary<string, object?> { ["lang"] = "en", ["year"] = 2024, ["tags"] = new[] { "a", "b" } };
        var doc = new Doc("readme", Readme, meta);
        var r = await docs.UpsertAsync(doc, EmbedAsync);
        Assert.True(r.Changed);
        Assert.Equal(r.Chunks, r.Embedded);
        var again = await docs.UpsertAsync(doc, EmbedAsync);
        Assert.False(again.Changed);
        Assert.Equal(0, again.Embedded);

        var q = Embed(["rust rust rust"])[0];
        var filter = new Dictionary<string, object?> { ["year"] = new Dictionary<string, object?> { ["$gte"] = 2020 } };
        var hits = docs.Search(q, new SearchOptions { K = 5, Filter = filter });
        Assert.Equal("readme", hits[0].Key);
        Assert.NotNull(hits[0].VectorScore);
        Assert.Equal("en", hits[0].Meta["lang"]);
        Assert.IsType<long>(hits[0].Meta["year"]);
        Assert.Equal(2024L, hits[0].Meta["year"]);
        Assert.Equal(new List<object?> { "a", "b" }, hits[0].Meta["tags"]);
        Assert.Empty(docs.Search(q, new SearchOptions { Filter = new Dictionary<string, object?> { ["tags"] = "z" } }));
        Assert.NotNull(docs.Search(q, new SearchOptions { Text = "Usage" })[0].LexicalScore);

        Assert.Single(docs.SearchDocuments(q, agg: "sum", top: 2));
        Assert.Contains("rust", docs.SearchSections(q)[0].Text);
        Assert.NotEmpty(docs.SearchExpand(q, 1));
        Assert.Equal(ErrorCode.Dimension, Assert.Throws<VdbliteException>(() => docs.Search([1, 2])).Code);

        var d = docs.Get("readme")!;
        Assert.StartsWith("# Intro", d.Content);
        Assert.Equal("markdown", d.Format);
        Assert.Null(docs.Get("missing"));
        Assert.Equal(r.Chunks, docs.Chunks("readme").Count);
        Assert.Single(docs.Scan(new Dictionary<string, object?> { ["lang"] = "en" }));
        var info = docs.Info();
        Assert.Equal(3, info.Dim);
        Assert.Equal("int8", info.Quantization);

        var inputs = docs.EmbedInputs("héllo wörld ✓", "text");
        var res = db.Write(WriteOp.Upsert(docs, new Doc("u", "héllo wörld ✓"), Embed(inputs)), WriteOp.Delete(docs, "readme"));
        Assert.True(res[1].Deleted);
        Assert.Equal("héllo wörld ✓", docs.Get("u")!.Content);
        Assert.Equal(1, docs.Count());

        using (var ro = Database.OpenReadOnly(path))
        {
            var rdocs = ro.Collection("docs");
            Assert.Equal(ErrorCode.ReadOnly, Assert.Throws<VdbliteException>(() => rdocs.Delete("u")).Code);
            ro.Close();
            Assert.Equal(ErrorCode.Closed, Assert.Throws<VdbliteException>(() => rdocs.Count()).Code);
        }
        Assert.Equal(ErrorCode.NoSuchCollection, Assert.Throws<VdbliteException>(() => db.Collection("nope")).Code);
    }

    [Fact]
    public void BulkLoadBuildsTheIndex()
    {
        using var db = Database.Open(Path.Combine(_dir, "bulk.vdb"));
        var c = db.CreateCollection("v", new CollectionOptions(2) { Quantization = "binary", Chunking = Chunking.None });
        var load = c.BulkLoad();
        var items = Enumerable.Range(0, 500)
            .Select(i => (new Doc($"k{i}", "doc"), (IReadOnlyList<float[]>)[new float[] { i % 7 + 1, 1 }]));
        Assert.Equal(500, load.Add(items));
        Assert.False(load.IsFinished);
        load.Finish();
        Assert.True(load.IsFinished);
        Assert.Equal(ErrorCode.InvalidArgument, Assert.Throws<VdbliteException>(load.Finish).Code);
        Assert.Equal(3, c.Search([7, 1], new SearchOptions { K = 3 }).Count);
        Assert.Equal(500, c.Stats().Chunks);
        Assert.True(BulkLoad.DefaultCheckpoint > 0);
    }
}
