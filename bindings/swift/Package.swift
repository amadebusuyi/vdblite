// swift-tools-version:5.9
import PackageDescription

// CVDBLite is the vdblite C library as an XCFramework: built locally by
// scripts/build-xcframework.sh; the release workflow publishes it and
// points this target at the download (url and checksum).
let package = Package(
    name: "VDBLite",
    platforms: [.macOS(.v12), .iOS(.v15)],
    products: [
        .library(name: "VDBLite", targets: ["VDBLite"]),
    ],
    targets: [
        .binaryTarget(name: "CVDBLite", path: "CVDBLite.xcframework"),
        .target(
            name: "VDBLite",
            dependencies: ["CVDBLite"],
            linkerSettings: [.linkedLibrary("iconv")]
        ),
        .testTarget(name: "VDBLiteTests", dependencies: ["VDBLite"]),
    ]
)
