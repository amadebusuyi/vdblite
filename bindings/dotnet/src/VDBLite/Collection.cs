namespace VDBLite;

/// <summary>A collection in an open database. Thread-safe.</summary>
public sealed class Collection
{
    private readonly CollectionHandle _h;

    internal Collection(Database db, CollectionHandle h, string name)
    {
        Database = db;
        _h = h;
        Name = name;
    }

    /// <summary>The database this collection is in.</summary>
    public Database Database { get; }

    /// <summary>The collection's name.</summary>
    public string Name { get; }

    /// <summary>The collection's settings.</summary>
    public CollectionInfo Info()
    {
        Native.Check(Native.vdb_collection_info(_h, out var json));
        return Json.Read<CollectionInfo>(json);
    }

    /// <summary>The state of the collection's index.</summary>
    public IndexStats Stats()
    {
        Native.Check(Native.vdb_collection_stats(_h, out var json));
        return Json.Read<IndexStats>(json);
    }

    /// <summary>The texts to embed for <paramref name="content"/>, one per chunk.</summary>
    public IReadOnlyList<string> EmbedInputs(string content, string? format = null)
    {
        Native.Check(Native.vdb_embed_inputs(_h, content, format, out var json));
        return Json.Read<List<string>>(json);
    }

    /// <summary>Inserts or replaces a document with one vector per chunk.</summary>
    public WriteResult Upsert(Doc doc, IReadOnlyList<float[]> vectors) =>
        Database.Write(WriteOp.Upsert(this, doc, vectors))[0];

    /// <summary>
    /// Inserts or replaces a document, calling <paramref name="embed"/> for the chunks that need vectors.
    /// An edited document re-embeds only the chunks whose text changed.
    /// </summary>
    public async Task<WriteResult> UpsertAsync(Doc doc, Embedder embed)
    {
        for (var attempt = 0; ; attempt++)
        {
            Native.Check(Native.vdb_plan_embeddings(_h, Json.Write(WriteOp.DocFields(doc)), out var json));
            var plan = Json.Read<List<PlannedChunk>>(json);
            IReadOnlyList<float[]> vectors = [];
            if (plan.Count > 0)
            {
                var texts = plan.Select(p => p.Text).ToList();
                try
                {
                    vectors = await embed(texts).ConfigureAwait(false);
                }
                catch (Exception e)
                {
                    throw new VdbliteException(ErrorCode.Embedder, e.Message);
                }
                if (vectors.Count != texts.Count)
                {
                    throw new VdbliteException(
                        ErrorCode.Embedder, $"the embedder returned {vectors.Count} vectors for {texts.Count} texts");
                }
            }
            try
            {
                return Database.Write(WriteOp.UpsertPlanned(this, doc, plan.Select(p => p.Hash).ToList(), vectors))[0];
            }
            catch (VdbliteException e) when (e.Code == ErrorCode.StaleEmbeddings && attempt < 3)
            {
                // Another writer changed the document's chunks after planning.
            }
        }
    }

    private sealed record PlannedChunk(string Hash, string Text);

    /// <summary>Deletes a document; true if it existed.</summary>
    public bool Delete(string key) => Database.Write(WriteOp.Delete(this, key))[0].Deleted;

    private T Search<T>(float[] query, SearchOptions? options, Dictionary<string, object?> extra)
    {
        var o = new Dictionary<string, object?>();
        if (options is not null)
        {
            if (options.K is { } k) o["k"] = k;
            if (options.Ef is { } ef) o["ef"] = ef;
            if (options.Filter is not null) o["filter"] = options.Filter;
            if (options.Text is not null) o["text"] = options.Text;
        }
        foreach (var (key, value) in extra) o[key] = value;
        Native.Check(Native.vdb_search(_h, query, (nuint)query.Length, Json.Write(o), out var json));
        return Json.Read<T>(json);
    }

    /// <summary>The chunks nearest to <paramref name="query"/>, best first.</summary>
    public IReadOnlyList<ChunkHit> Search(float[] query, SearchOptions? options = null) =>
        Search<List<ChunkHit>>(query, options, new() { ["mode"] = "chunks" });

    /// <summary>
    /// Documents ranked by their best chunk (<paramref name="agg"/> "max") or by the sum of their
    /// <paramref name="top"/> best chunks ("sum").
    /// </summary>
    public IReadOnlyList<DocHit> SearchDocuments(float[] query, SearchOptions? options = null, string agg = "max", int top = 3) =>
        Search<List<DocHit>>(query, options, new() { ["mode"] = "documents", ["agg"] = agg, ["top"] = top });

    /// <summary>Hits widened by <paramref name="neighbors"/> chunks on each side, with overlaps merged.</summary>
    public IReadOnlyList<Passage> SearchExpand(float[] query, int neighbors = 1, SearchOptions? options = null) =>
        Search<List<Passage>>(query, options, new() { ["mode"] = "expand", ["neighbors"] = neighbors });

    /// <summary>The whole heading section around each hit.</summary>
    public IReadOnlyList<Passage> SearchSections(float[] query, SearchOptions? options = null) =>
        Search<List<Passage>>(query, options, new() { ["mode"] = "sections" });

    /// <summary>The document with <paramref name="key"/>, or null.</summary>
    public Document? Get(string key)
    {
        Native.Check(Native.vdb_get(_h, key, out var json));
        return Json.Read<Document?>(json);
    }

    /// <summary>The chunks of the document with <paramref name="key"/>.</summary>
    public IReadOnlyList<ChunkInfo> Chunks(string key)
    {
        Native.Check(Native.vdb_chunks(_h, key, out var json));
        return Json.Read<List<ChunkInfo>>(json);
    }

    /// <summary>Summaries of the documents matching <paramref name="filter"/> (null: all).</summary>
    public IReadOnlyList<DocInfo> Scan(IReadOnlyDictionary<string, object?>? filter = null)
    {
        Native.Check(Native.vdb_scan(_h, filter is null ? null : Json.Write(filter), out var json));
        return Json.Read<List<DocInfo>>(json);
    }

    /// <summary>The number of documents.</summary>
    public long Count()
    {
        Native.Check(Native.vdb_count(_h, out var n));
        return (long)n;
    }

    /// <summary>Whether the collection keeps a keyword index for hybrid search.</summary>
    public bool HasLexical()
    {
        Native.Check(Native.vdb_has_lexical(_h, out var has));
        return has;
    }

    /// <summary>Builds the keyword index from stored text, without re-embedding; false if it existed.</summary>
    public bool EnableLexical()
    {
        Native.Check(Native.vdb_enable_lexical(_h, out var built));
        return built;
    }

    /// <summary>
    /// Starts a bulk load: added documents are readable at once and searchable after
    /// <see cref="BulkLoad.Finish"/>, which builds the index once with every core.
    /// </summary>
    /// <param name="checkpointEvery">Chunks between durable commits in a file opened exclusively (0: <see cref="BulkLoad.DefaultCheckpoint"/>).</param>
    public BulkLoad BulkLoad(ulong checkpointEvery = 0)
    {
        Native.Check(Native.vdb_bulk_load(_h, checkpointEvery, out var b));
        return new BulkLoad(this, b);
    }
}
