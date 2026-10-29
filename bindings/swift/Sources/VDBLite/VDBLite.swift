// VDBLite: an embedded, single-file vector database. Open a file, store
// documents, and search them by meaning, with no server.
//
// Calls block (the heavy ones use every core); call them off the main
// actor. Every type is thread-safe.

import CVDBLite
import Foundation

// MARK: - Errors

/// The kind of a ``VdbliteError``.
public enum ErrorCode: Int32, Sendable {
    /// The file or a transaction failed, or stored bytes were invalid.
    case storage = 1
    /// A vector had the wrong number of dimensions.
    case dimension = 2
    case collectionExists = 3
    case noSuchCollection = 4
    case invalidArgument = 5
    /// The embedder failed.
    case embedder = 6
    /// The document changed after its embeddings were planned.
    case staleEmbeddings = 7
    /// The file's format version isn't supported by this build.
    case version = 8
    /// Sealing or compacting the index failed; nothing committed was lost.
    case maintenance = 9
    case readOnly = 10
    case closed = 11
    /// A bug in vdblite: please report it.
    case panic = 12
}

/// A failed vdblite call.
public struct VdbliteError: Error, CustomStringConvertible, Sendable {
    public let code: ErrorCode
    public let message: String

    public var description: String { "\(code): \(message)" }
}

@inline(__always)
private func check(_ status: Int32) throws {
    if status != VDB_OK {
        throw VdbliteError(code: ErrorCode(rawValue: status) ?? .storage, message: String(cString: vdb_last_error()))
    }
}

// MARK: - JSON values

/// A JSON value: metadata values (strings, numbers, booleans, null, lists)
/// and filters, written as literals:
///
///     let filter: Filter = ["lang": "en", "year": ["$gte": 2020], "tags": ["$in": ["a", "b"]]]
public enum JSONValue: Hashable, Sendable, Codable,
    ExpressibleByStringLiteral, ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral,
    ExpressibleByBooleanLiteral, ExpressibleByNilLiteral, ExpressibleByArrayLiteral,
    ExpressibleByDictionaryLiteral
{
    case null
    case bool(Bool)
    case int(Int64)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(stringLiteral value: String) { self = .string(value) }
    public init(integerLiteral value: Int64) { self = .int(value) }
    public init(floatLiteral value: Double) { self = .double(value) }
    public init(booleanLiteral value: Bool) { self = .bool(value) }
    public init(nilLiteral: ()) { self = .null }
    public init(arrayLiteral elements: JSONValue...) { self = .array(elements) }
    public init(dictionaryLiteral elements: (String, JSONValue)...) {
        self = .object(Dictionary(elements, uniquingKeysWith: { _, last in last }))
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self = .null
        } else if let b = try? c.decode(Bool.self) {
            self = .bool(b)
        } else if let i = try? c.decode(Int64.self) {
            self = .int(i)
        } else if let d = try? c.decode(Double.self) {
            self = .double(d)
        } else if let s = try? c.decode(String.self) {
            self = .string(s)
        } else if let a = try? c.decode([JSONValue].self) {
            self = .array(a)
        } else {
            self = .object(try c.decode([String: JSONValue].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .int(let i): try c.encode(i)
        case .double(let d): try c.encode(d)
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }
}

/// A document's metadata.
public typealias Meta = [String: JSONValue]

/// A Mongo-style metadata filter. Operators: $eq $ne $in $nin $gt $gte $lt
/// $lte $exists $and $or $not. Equality on a list field matches any element.
public typealias Filter = [String: JSONValue]

private enum Coding {
    static let encoder = JSONEncoder()
    static let decoder = JSONDecoder()

    static func string<T: Encodable>(_ value: T) throws -> String {
        String(decoding: try encoder.encode(value), as: UTF8.self)
    }

    /// Calls `f` for a JSON string from the library, decodes it, and frees it.
    static func take<T: Decodable>(_ type: T.Type, _ f: (UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> Int32) throws -> T {
        var out: UnsafeMutablePointer<CChar>?
        try check(f(&out))
        defer { vdb_string_free(out) }
        return try decoder.decode(T.self, from: Data(String(cString: out!).utf8))
    }
}

// MARK: - Inputs

/// Turns texts into vectors, one per text, in order: any embedding model.
public typealias Embedder = @Sendable ([String]) async throws -> [[Float]]

/// A document to write.
public struct Doc: Sendable, Encodable {
    /// Unique within the collection.
    public var key: String
    public var content: String
    public var meta: Meta?
    /// "markdown", "code", "html", "text", a file name, extension or MIME
    /// type. Sniffed if nil.
    public var format: String?

    public init(key: String, content: String, meta: Meta? = nil, format: String? = nil) {
        self.key = key
        self.content = content
        self.meta = meta
        self.format = format
    }
}

/// How documents are split into chunks.
public enum Chunking: Sendable, Encodable {
    /// Along headings, paragraphs, code blocks and definitions.
    case structural(targetTokens: Int = 400, overlapTokens: Int = 50)
    /// Each document is one chunk with one vector.
    case none

    private enum Keys: String, CodingKey { case kind, targetTokens, overlapTokens }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: Keys.self)
        switch self {
        case .structural(let target, let overlap):
            try c.encode("structural", forKey: .kind)
            try c.encode(target, forKey: .targetTokens)
            try c.encode(overlap, forKey: .overlapTokens)
        case .none:
            try c.encode("none", forKey: .kind)
        }
    }
}

/// Options for a new collection. Only `dim` is required.
public struct CollectionOptions: Sendable, Encodable {
    public var dim: Int
    /// "cosine" (default), "dot" or "l2".
    public var metric: String?
    /// "int8" (default: int8 graph walk, f16 re-ranking), "binary" (1-bit
    /// graph walk, f16 re-ranking) or "none" (f32).
    public var quantization: String?
    public var chunking: Chunking?
    /// Chunks buffered before they're sealed into a graph (default 65536).
    public var sealThreshold: Int?
    /// Keep a BM25 keyword index for hybrid search (default true).
    public var lexical: Bool?
    public var m: Int?
    public var efConstruction: Int?
    public var efSearch: Int?

    public init(
        dim: Int, metric: String? = nil, quantization: String? = nil, chunking: Chunking? = nil,
        sealThreshold: Int? = nil, lexical: Bool? = nil, m: Int? = nil, efConstruction: Int? = nil,
        efSearch: Int? = nil
    ) {
        self.dim = dim
        self.metric = metric
        self.quantization = quantization
        self.chunking = chunking
        self.sealThreshold = sealThreshold
        self.lexical = lexical
        self.m = m
        self.efConstruction = efConstruction
        self.efSearch = efSearch
    }
}

/// Refines a search.
public struct SearchOptions: Sendable {
    /// Results to return.
    public var k: Int
    /// Graph beam width: higher is slower and more accurate.
    public var ef: Int?
    public var filter: Filter?
    /// Query text for hybrid search: chunks are also ranked by BM25 keyword
    /// match and the rankings fused.
    public var text: String?

    public init(k: Int = 10, ef: Int? = nil, filter: Filter? = nil, text: String? = nil) {
        self.k = k
        self.ef = ef
        self.filter = filter
        self.text = text
    }
}

// MARK: - Results

/// A chunk that matched a search. `byteStart`/`byteEnd` are UTF-8 byte
/// offsets in the document. `score`: higher is better (cosine similarity,
/// dot product or negative squared L2; in a hybrid search the fused rank
/// score). `vectorScore` is nil if only the keyword search found the chunk,
/// `lexicalScore` if the keyword search didn't match it.
public struct ChunkHit: Sendable, Decodable {
    public let key: String
    public let docId: UInt64
    public let chunkId: UInt32
    public let ordinal: UInt32
    public let byteStart: Int
    public let byteEnd: Int
    public let score: Float
    public let vectorScore: Float?
    public let lexicalScore: Float?
    public let text: String
    public let headingPath: [String]
    public let meta: Meta
}

/// A document ranked by its chunks; `chunks` are its matching chunks, best first.
public struct DocHit: Sendable, Decodable {
    public let key: String
    public let docId: UInt64
    public let score: Float
    public let meta: Meta
    public let chunks: [ChunkHit]
}

/// A hit widened to neighbouring chunks or its section; `ordinals` are the
/// chunks covered, [start, end).
public struct Passage: Sendable, Decodable {
    public let key: String
    public let docId: UInt64
    public let byteStart: Int
    public let byteEnd: Int
    public let score: Float
    public let text: String
    public let ordinals: [UInt32]
    public let headingPath: [String]
    public let meta: Meta
}

/// A stored document with its content.
public struct Document: Sendable, Decodable {
    public let key: String
    public let docId: UInt64
    public let content: String
    public let meta: Meta
    public let version: UInt64
    public let createdMs: UInt64
    public let updatedMs: UInt64
    /// "markdown", "code", "html" or "text".
    public let format: String
    public let chunkCount: Int
}

/// A summary of a stored document.
public struct DocInfo: Sendable, Decodable {
    public let key: String
    public let docId: UInt64
    public let meta: Meta
    public let version: UInt64
    public let createdMs: UInt64
    public let updatedMs: UInt64
    public let contentLength: Int
    public let chunkCount: Int
}

/// One chunk of a stored document.
public struct ChunkInfo: Sendable, Decodable {
    public let chunkId: UInt32
    public let ordinal: UInt32
    public let byteStart: Int
    public let byteEnd: Int
    public let overlapStart: Int
    public let headingPath: [String]
    public let tokens: Int
}

/// The outcome of one write. For an upsert, `embedded` counts the chunks
/// that took new vectors and `changed` is false when nothing changed; for a
/// delete, `deleted` says whether the document existed.
public struct WriteResult: Sendable, Decodable {
    public let op: String
    public let collection: String
    public let key: String
    public let docId: UInt64?
    public let version: UInt64?
    public let chunks: Int?
    public let embedded: Int?
    public let changed: Bool?
    public let deleted: Bool?
}

/// A collection's settings.
public struct CollectionInfo: Sendable, Decodable {
    public let name: String
    public let dim: Int
    public let metric: String
    public let quantization: String
    public let chunking: String
    public let targetTokens: Int?
    public let overlapTokens: Int?
}

/// A collection's index; `maintenanceError` is the last background
/// maintenance failure.
public struct IndexStats: Sendable, Decodable {
    public let chunks: Int
    public let sealedSegments: Int
    public let buffered: Int
    public let memoryBytes: Int
    public let maintenanceError: String?
}

// MARK: - Database

/// An open vdblite database file. One process writes a file at a time;
/// other processes can open it read-only alongside and see each commit as
/// it lands.
///
///     let db = try Database.open(path)
///     let docs = try db.getOrCreateCollection("docs", options: CollectionOptions(dim: 768))
///     try await docs.upsert(Doc(key: "readme", content: markdown), embed: embedder)
///     let hits = try docs.search(queryVector, options: SearchOptions(k: 5))
public final class Database: @unchecked Sendable {
    let h: OpaquePointer

    private init(_ h: OpaquePointer) { self.h = h }

    deinit { vdb_db_free(h) }

    /// The version of the native library.
    public static var version: String { String(cString: vdb_version()) }

    /// Opens (or creates) a database file. `readOnly` follows another
    /// process's writes; `exclusive` shuts every other process out for
    /// faster small commits. `cacheBytes`: 0 for the default (256 MiB).
    public static func open(
        _ path: String, readOnly: Bool = false, exclusive: Bool = false, cacheBytes: UInt64 = 0
    ) throws -> Database {
        var flags: UInt32 = 0
        if readOnly { flags |= UInt32(VDB_OPEN_READ_ONLY) }
        if exclusive { flags |= UInt32(VDB_OPEN_EXCLUSIVE) }
        var h: OpaquePointer?
        try check(vdb_open(path, flags, cacheBytes, &h))
        return Database(h!)
    }

    /// Rewrites a file nobody has open without its free space; true if it shrank.
    @discardableResult
    public static func compactFile(_ path: String) throws -> Bool {
        var shrank = false
        try check(vdb_compact_file(path, &shrank))
        return shrank
    }

    /// Closes the database; later calls on it and its collections throw
    /// ``ErrorCode/closed``.
    public func close() { vdb_close(h) }

    public func isReadOnly() throws -> Bool {
        var out = false
        try check(vdb_is_read_only(h, &out))
        return out
    }

    public func collections() throws -> [String] {
        try Coding.take([String].self) { vdb_collections(h, $0) }
    }

    public func createCollection(_ name: String, options: CollectionOptions) throws -> Collection {
        var c: OpaquePointer?
        try check(vdb_create_collection(h, name, try Coding.string(options), &c))
        return Collection(db: self, h: c!, name: name)
    }

    /// An existing collection; throws ``ErrorCode/noSuchCollection`` if there is none.
    public func collection(_ name: String) throws -> Collection {
        var c: OpaquePointer?
        try check(vdb_get_collection(h, name, &c))
        return Collection(db: self, h: c!, name: name)
    }

    public func getOrCreateCollection(_ name: String, options: CollectionOptions) throws -> Collection {
        do {
            return try collection(name)
        } catch let e as VdbliteError where e.code == .noSuchCollection {
            return try createCollection(name, options: options)
        }
    }

    public func dropCollection(_ name: String) throws {
        try check(vdb_drop_collection(h, name))
    }

    /// Applies upserts and deletes atomically, in one transaction.
    @discardableResult
    public func write(_ ops: [WriteOp]) throws -> [WriteResult] {
        struct Response: Decodable {
            let results: [WriteResult]
            let warning: String?
        }
        let flat = ops.flatMap { $0.vectors.flatMap { $0 } }
        let json = try Coding.string(ops)
        let res = try flat.withUnsafeBufferPointer { buf in
            try Coding.take(Response.self) { vdb_write(h, json, buf.baseAddress, buf.count, $0) }
        }
        if let warning = res.warning {
            // Committed; only sealing the write buffer afterwards failed.
            throw VdbliteError(code: .maintenance, message: warning)
        }
        return res.results
    }

    /// Seals buffered chunks and merges small segments in every collection,
    /// using every core.
    public func optimize() throws { try check(vdb_optimize(h)) }

    /// Waits for background sealing and compaction to catch up with every
    /// commit so far.
    public func waitForMaintenance() throws { try check(vdb_wait_for_maintenance(h)) }
}

/// One change for ``Database/write(_:)``.
public struct WriteOp: Sendable, Encodable {
    let op: String
    let collection: String
    let doc: Doc
    let vectors: [[Float]]
    let hashes: [String]?

    /// Inserts or replaces a document with one vector per chunk (see
    /// ``Collection/embedInputs(_:format:)``).
    public static func upsert(_ collection: Collection, _ doc: Doc, vectors: [[Float]]) -> WriteOp {
        WriteOp(op: "upsert", collection: collection.name, doc: doc, vectors: vectors, hashes: nil)
    }

    public static func delete(_ collection: Collection, key: String) -> WriteOp {
        WriteOp(op: "delete", collection: collection.name, doc: Doc(key: key, content: ""), vectors: [], hashes: nil)
    }

    private enum Keys: String, CodingKey { case op, collection, key, content, meta, format, vectors, embeddings }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: Keys.self)
        try c.encode(op, forKey: .op)
        try c.encode(collection, forKey: .collection)
        try c.encode(doc.key, forKey: .key)
        guard op == "upsert" else { return }
        try c.encode(doc.content, forKey: .content)
        try c.encodeIfPresent(doc.meta, forKey: .meta)
        try c.encodeIfPresent(doc.format, forKey: .format)
        if let hashes {
            try c.encode(hashes, forKey: .embeddings)
        } else {
            try c.encode(vectors.count, forKey: .vectors)
        }
    }
}

// MARK: - Collection

/// A collection in an open database.
public final class Collection: @unchecked Sendable {
    /// Keeps the database handle alive while this collection is.
    public let database: Database
    let h: OpaquePointer
    public let name: String

    init(db: Database, h: OpaquePointer, name: String) {
        database = db
        self.h = h
        self.name = name
    }

    deinit { vdb_collection_free(h) }

    public func info() throws -> CollectionInfo {
        try Coding.take(CollectionInfo.self) { vdb_collection_info(h, $0) }
    }

    public func stats() throws -> IndexStats {
        try Coding.take(IndexStats.self) { vdb_collection_stats(h, $0) }
    }

    /// The texts to embed for `content`, one per chunk.
    public func embedInputs(_ content: String, format: String? = nil) throws -> [String] {
        try Coding.take([String].self) { out in
            if let format {
                return vdb_embed_inputs(h, content, format, out)
            }
            return vdb_embed_inputs(h, content, nil, out)
        }
    }

    /// Inserts or replaces a document with one vector per chunk.
    @discardableResult
    public func upsert(_ doc: Doc, vectors: [[Float]]) throws -> WriteResult {
        try database.write([.upsert(self, doc, vectors: vectors)])[0]
    }

    /// Inserts or replaces a document, calling `embed` for the chunks that
    /// need vectors. An edited document re-embeds only the chunks whose
    /// text changed.
    @discardableResult
    public func upsert(_ doc: Doc, embed: Embedder) async throws -> WriteResult {
        struct Planned: Decodable {
            let hash: String
            let text: String
        }
        var attempt = 0
        while true {
            let json = try Coding.string(doc)
            let plan = try Coding.take([Planned].self) { vdb_plan_embeddings(h, json, $0) }
            var vectors: [[Float]] = []
            if !plan.isEmpty {
                do {
                    vectors = try await embed(plan.map(\.text))
                } catch {
                    throw VdbliteError(code: .embedder, message: "\(error)")
                }
                if vectors.count != plan.count {
                    throw VdbliteError(
                        code: .embedder, message: "the embedder returned \(vectors.count) vectors for \(plan.count) texts")
                }
            }
            let op = WriteOp(op: "upsert", collection: name, doc: doc, vectors: vectors, hashes: plan.map(\.hash))
            do {
                return try database.write([op])[0]
            } catch let e as VdbliteError where e.code == .staleEmbeddings && attempt < 3 {
                // Another writer changed the document's chunks after planning.
                attempt += 1
            }
        }
    }

    /// Deletes a document; true if it existed.
    @discardableResult
    public func delete(_ key: String) throws -> Bool {
        try database.write([.delete(self, key: key)])[0].deleted ?? false
    }

    private struct SearchJSON: Encodable {
        let mode: String
        let k: Int
        let ef: Int?
        let filter: Filter?
        let text: String?
        var agg: String?
        var top: Int?
        var neighbors: Int?
    }

    private func search<T: Decodable>(_ query: [Float], _ o: SearchOptions, _ mode: String, _ edit: (inout SearchJSON) -> Void = { _ in }) throws -> T {
        var s = SearchJSON(mode: mode, k: o.k, ef: o.ef, filter: o.filter, text: o.text)
        edit(&s)
        let json = try Coding.string(s)
        return try query.withUnsafeBufferPointer { q in
            try Coding.take(T.self) { vdb_search(h, q.baseAddress, q.count, json, $0) }
        }
    }

    /// The chunks nearest to `query`, best first.
    public func search(_ query: [Float], options: SearchOptions = SearchOptions()) throws -> [ChunkHit] {
        try search(query, options, "chunks")
    }

    /// Documents ranked by their best chunk (`agg` "max") or by the sum of
    /// their `top` best chunks ("sum").
    public func searchDocuments(
        _ query: [Float], options: SearchOptions = SearchOptions(), agg: String = "max", top: Int = 3
    ) throws -> [DocHit] {
        try search(query, options, "documents") {
            $0.agg = agg
            $0.top = top
        }
    }

    /// Hits widened by `neighbors` chunks on each side, with overlaps merged.
    public func searchExpand(_ query: [Float], neighbors: Int = 1, options: SearchOptions = SearchOptions()) throws -> [Passage] {
        try search(query, options, "expand") { $0.neighbors = neighbors }
    }

    /// The whole heading section around each hit.
    public func searchSections(_ query: [Float], options: SearchOptions = SearchOptions()) throws -> [Passage] {
        try search(query, options, "sections")
    }

    /// The document with `key`, or nil.
    public func get(_ key: String) throws -> Document? {
        try Coding.take(Document?.self) { vdb_get(h, key, $0) }
    }

    public func chunks(_ key: String) throws -> [ChunkInfo] {
        try Coding.take([ChunkInfo].self) { vdb_chunks(h, key, $0) }
    }

    /// Summaries of the documents matching `filter` (nil: all).
    public func scan(_ filter: Filter? = nil) throws -> [DocInfo] {
        let json = try filter.map { try Coding.string($0) }
        return try Coding.take([DocInfo].self) { out in
            if let json {
                return vdb_scan(h, json, out)
            }
            return vdb_scan(h, nil, out)
        }
    }

    public func count() throws -> Int {
        var n: UInt64 = 0
        try check(vdb_count(h, &n))
        return Int(n)
    }

    public func hasLexical() throws -> Bool {
        var out = false
        try check(vdb_has_lexical(h, &out))
        return out
    }

    /// Builds the keyword index from stored text, without re-embedding;
    /// false if it existed.
    @discardableResult
    public func enableLexical() throws -> Bool {
        var built = false
        try check(vdb_enable_lexical(h, &built))
        return built
    }

    /// Starts a bulk load: added documents are readable at once and
    /// searchable after ``BulkLoad/finish()``, which builds the index once
    /// with every core. `checkpointEvery`: chunks between durable commits
    /// in a file opened exclusively (0 for ``BulkLoad/defaultCheckpoint``).
    public func bulkLoad(checkpointEvery: UInt64 = 0) throws -> BulkLoad {
        var b: OpaquePointer?
        try check(vdb_bulk_load(h, checkpointEvery, &b))
        return BulkLoad(collection: self, h: b!)
    }
}

// MARK: - Bulk load

/// A bulk load in progress. Always call ``finish()``: until then, loaded
/// documents can be read but not found by search.
public final class BulkLoad: @unchecked Sendable {
    /// Chunks between durable commits, by default.
    public static let defaultCheckpoint = vdb_default_checkpoint()

    public let collection: Collection
    let h: OpaquePointer

    init(collection: Collection, h: OpaquePointer) {
        self.collection = collection
        self.h = h
    }

    deinit { vdb_bulk_free(h) }

    /// Inserts or replaces documents (each with one vector per chunk) in one
    /// transaction; returns how many were written.
    @discardableResult
    public func add(_ docs: [(doc: Doc, vectors: [[Float]])]) throws -> Int {
        struct Item: Encodable {
            let key: String
            let content: String
            let meta: Meta?
            let format: String?
            let vectors: Int
        }
        let items = docs.map {
            Item(key: $0.doc.key, content: $0.doc.content, meta: $0.doc.meta, format: $0.doc.format, vectors: $0.vectors.count)
        }
        let json = try Coding.string(items)
        let flat = docs.flatMap { $0.vectors.flatMap { $0 } }
        var written: UInt64 = 0
        try flat.withUnsafeBufferPointer { buf in
            try check(vdb_bulk_add(h, json, buf.baseAddress, buf.count, &written))
        }
        return Int(written)
    }

    /// Makes everything added so far durable (only matters in a file opened
    /// exclusively).
    public func checkpoint() throws { try check(vdb_bulk_checkpoint(h)) }

    /// Builds the index over everything added and makes it searchable.
    /// Other writes wait until it's done; reads and searches don't.
    public func finish() throws { try check(vdb_bulk_finish(h)) }

    public var isFinished: Bool {
        var out = false
        _ = vdb_bulk_finished(h, &out)
        return out
    }
}
