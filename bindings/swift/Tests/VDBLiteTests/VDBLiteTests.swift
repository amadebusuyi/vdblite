import Foundation
import Testing

@testable import VDBLite

/// A toy embedder: [length, count of "rust", 1].
@Sendable func embed(_ texts: [String]) -> [[Float]] {
    texts.map { [Float($0.count), Float($0.components(separatedBy: "rust").count - 1), 1] }
}

func scratch(_ name: String) -> String {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("vdblite-swift-\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir.appendingPathComponent(name).path
}

let readme = "# Intro\n\nrust rust rust\n\n## Usage\n\nOther words here."

@Test func roundTrip() async throws {
    let path = scratch("test.vdb")
    let db = try Database.open(path)
    #expect(!Database.version.isEmpty)
    let docs = try db.createCollection("docs", options: CollectionOptions(dim: 3))
    #expect(throws: VdbliteError.self) { try db.createCollection("docs", options: CollectionOptions(dim: 3)) }

    let doc = Doc(key: "readme", content: readme, meta: ["lang": "en", "year": 2024, "tags": ["a", "b"]])
    let r = try await docs.upsert(doc, embed: { embed($0) })
    #expect(r.changed == true && r.embedded == r.chunks)
    let again = try await docs.upsert(doc, embed: { embed($0) })
    #expect(again.changed == false && again.embedded == 0)

    let q = embed(["rust rust rust"])[0]
    let hits = try docs.search(q, options: SearchOptions(k: 5, filter: ["year": ["$gte": 2020]]))
    #expect(hits.first?.key == "readme")
    #expect(hits.first?.vectorScore != nil)
    #expect(hits.first?.meta["year"] == .int(2024))
    #expect(hits.first?.meta["tags"] == ["a", "b"])
    #expect(try docs.search(q, options: SearchOptions(filter: ["tags": "z"])).isEmpty)
    #expect(try docs.search(q, options: SearchOptions(text: "Usage")).first?.lexicalScore != nil)
    #expect(try docs.searchDocuments(q, agg: "sum", top: 2).count == 1)
    #expect(try docs.searchSections(q).first?.text.contains("rust") == true)
    #expect(try !docs.searchExpand(q, neighbors: 1).isEmpty)
    do {
        _ = try docs.search([1, 2])
        Issue.record("expected a dimension error")
    } catch let e as VdbliteError {
        #expect(e.code == .dimension)
    }

    let d = try #require(try docs.get("readme"))
    #expect(d.content.hasPrefix("# Intro") && d.format == "markdown")
    #expect(try docs.get("missing") == nil)
    #expect(try docs.chunks("readme").count == r.chunks)
    #expect(try docs.scan(["lang": "en"]).count == 1)
    let info = try docs.info()
    #expect(info.dim == 3 && info.quantization == "int8")

    let inputs = try docs.embedInputs("héllo wörld ✓", format: "text")
    let res = try db.write([.upsert(docs, Doc(key: "u", content: "héllo wörld ✓"), vectors: embed(inputs)), .delete(docs, key: "readme")])
    #expect(res[1].deleted == true)
    #expect(try docs.get("u")?.content == "héllo wörld ✓")
    #expect(try docs.count() == 1)

    let ro = try Database.open(path, readOnly: true)
    let rdocs = try ro.collection("docs")
    do {
        try rdocs.delete("u")
        Issue.record("expected a read-only error")
    } catch let e as VdbliteError {
        #expect(e.code == .readOnly)
    }
    ro.close()
    do {
        _ = try rdocs.count()
        Issue.record("expected a closed error")
    } catch let e as VdbliteError {
        #expect(e.code == .closed)
    }
    db.close()
}

@Test func bulkLoad() throws {
    let db = try Database.open(scratch("bulk.vdb"))
    let c = try db.createCollection("v", options: CollectionOptions(dim: 2, quantization: "binary", chunking: Chunking.none))
    let load = try c.bulkLoad()
    let items = (0..<500).map { i in (doc: Doc(key: "k\(i)", content: "doc"), vectors: [[Float(i % 7 + 1), 1]]) }
    #expect(try load.add(items) == 500)
    #expect(!load.isFinished)
    try load.finish()
    #expect(load.isFinished)
    #expect(try c.search([7, 1], options: SearchOptions(k: 3)).count == 3)
    #expect(try c.stats().chunks == 500)
    #expect(BulkLoad.defaultCheckpoint > 0)
    db.close()
}
