#!/usr/bin/env bash
# Installs a just-published VDBLite package from its registry, the way a user
# would (no Rust, no checkout of the bindings), then opens a file, writes and
# searches. Retries while the registry catches up.
# Usage: VERSION=0.1.0 scripts/release/smoke-lang.sh go|java|dotnet|swift
set -euo pipefail
lang="$1"
root=$(cd "$(dirname "$0")/../.." && pwd)
app="${RUNNER_TEMP:-$(mktemp -d)}/smoke-$lang"
rm -rf "$app" && mkdir -p "$app" && cd "$app"

retry() {
  for i in 1 2 3 4 5 6 7 8 9 10; do
    "$@" && return 0
    echo "attempt $i failed; retrying in 30s"
    sleep 30
  done
  return 1
}

case "$lang" in
  go)
    module=$(sed -n 's/^module //p' "$root/bindings/go/go.mod")
    go mod init smoke > /dev/null
    retry go get "$module@v$VERSION"
    cat > main.go <<EOF
package main

import (
	"fmt"
	"os"
	"path/filepath"

	vdblite "$module"
)

func main() {
	dir, err := os.MkdirTemp("", "smoke")
	if err != nil {
		panic(err)
	}
	db, err := vdblite.Open(filepath.Join(dir, "smoke.vdb"), nil)
	if err != nil {
		panic(err)
	}
	defer db.Close()
	c, err := db.CreateCollection("smoke", vdblite.CollectionOptions{Dim: 2})
	if err != nil {
		panic(err)
	}
	if _, err := c.Upsert(vdblite.Doc{Key: "a", Content: "hello"}, [][]float32{{1, 0}}); err != nil {
		panic(err)
	}
	hits, err := c.Search([]float32{1, 0}, nil)
	if err != nil || len(hits) != 1 {
		panic(fmt.Sprint(hits, err))
	}
	fmt.Println("go: found", hits[0].Key, "with vdblite", vdblite.Version())
}
EOF
    go run .
    ;;
  java)
    group=$(sed -n 's|^  <groupId>\(.*\)</groupId>|\1|p' "$root/bindings/java/pom.xml" | head -1)
    jna=$(grep -A1 '<artifactId>jna</artifactId>' "$root/bindings/java/pom.xml" | sed -n 's|.*<version>\(.*\)</version>|\1|p')
    retry mvn -B -q dependency:copy -Dartifact="$group:vdblite:$VERSION" -DoutputDirectory=lib
    mvn -B -q dependency:copy -Dartifact="net.java.dev.jna:jna:$jna" -DoutputDirectory=lib
    cat > Smoke.java <<'EOF'
import io.vdblite.*;
import java.nio.file.Files;
import java.util.List;

public class Smoke {
  public static void main(String[] args) throws Exception {
    try (Database db = Database.open(Files.createTempDirectory("smoke").resolve("smoke.vdb"))) {
      Collection c = db.createCollection("smoke", CollectionOptions.dim(2));
      c.upsert(new Doc("a", "hello"), List.of(new float[] {1, 0}));
      List<ChunkHit> hits = c.search(new float[] {1, 0});
      if (hits.size() != 1) throw new AssertionError(hits);
      System.out.println("java: found " + hits.get(0).key() + " with vdblite " + Database.version());
    }
  }
}
EOF
    java -cp "lib/*" Smoke.java
    ;;
  dotnet)
    dotnet new console -o . --force > /dev/null
    retry dotnet add package VDBLite --version "$VERSION"
    cat > Program.cs <<'EOF'
using VDBLite;

var dir = Directory.CreateTempSubdirectory("smoke").FullName;
using var db = Database.Open(Path.Combine(dir, "smoke.vdb"));
var c = db.CreateCollection("smoke", new CollectionOptions(2));
c.Upsert(new Doc("a", "hello"), [new float[] { 1, 0 }]);
var hits = c.Search([1, 0]);
if (hits.Count != 1) throw new Exception($"{hits.Count} hits");
Console.WriteLine($"dotnet: found {hits[0].Key} with vdblite {Database.Version}");
EOF
    dotnet run
    ;;
  swift)
    repo="https://github.com/${SWIFT_REPO:-$GITHUB_REPOSITORY_OWNER/vdblite-swift}"
    mkdir -p Sources/Smoke
    cat > Package.swift <<EOF
// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Smoke",
    platforms: [.macOS(.v12)],
    dependencies: [.package(url: "$repo", exact: "$VERSION")],
    targets: [.executableTarget(name: "Smoke", dependencies: [.product(name: "VDBLite", package: "vdblite-swift")])]
)
EOF
    cat > Sources/Smoke/main.swift <<'EOF'
import Foundation
import VDBLite

let path = FileManager.default.temporaryDirectory.appendingPathComponent("smoke-\(UUID().uuidString).vdb").path
let db = try Database.open(path)
let c = try db.createCollection("smoke", options: CollectionOptions(dim: 2))
try c.upsert(Doc(key: "a", content: "hello"), vectors: [[1, 0]])
let hits = try c.search([1, 0])
precondition(hits.count == 1)
print("swift: found \(hits[0].key) with vdblite \(Database.version)")
EOF
    retry swift package resolve
    swift run
    ;;
  *)
    echo "usage: smoke-lang.sh go|java|dotnet|swift" >&2
    exit 2
    ;;
esac
