#!/usr/bin/env bash
# Builds the vdblite C library as CVDBLite.xcframework: macOS (arm64 +
# x86_64), iOS devices, and the iOS simulator (arm64 + x86_64).
#   --host   only this Mac's architecture, for local development
#   --debug  a debug build
#   --zip    also write CVDBLite.xcframework.zip and print its checksum
# Needs Xcode for xcodebuild, except with --host.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$here/../.." && pwd)
profile=release-ffi dir=release-ffi host="" zip=""
for a in "$@"; do
  case "$a" in
    --host) host=1 ;;
    --debug) profile=dev dir=debug ;;
    --zip) zip=1 ;;
    *) echo "unknown argument $a" >&2; exit 2 ;;
  esac
done
cd "$root"
# Link on older systems than the one building.
export MACOSX_DEPLOYMENT_TARGET=12.0 IPHONEOS_DEPLOYMENT_TARGET=15.0
work="$root/target/xcframework"
rm -rf "$work" "$here/CVDBLite.xcframework"
mkdir -p "$work/headers"
cp "$root/bindings/c/include/vdblite.h" "$work/headers/"
cat > "$work/headers/module.modulemap" <<'MAP'
module CVDBLite {
    header "vdblite.h"
    link "vdblite"
    export *
}
MAP

lib() { cargo build -p vdblite-c --profile "$profile" --target "$1" >&2; echo "$root/target/$1/$dir/libvdblite.a"; }

if [ -n "$host" ]; then
  # Assembled by hand, so it works without Xcode.
  arch=$(uname -m)
  case "$arch" in arm64) triple=aarch64-apple-darwin ;; *) triple=x86_64-apple-darwin ;; esac
  a=$(lib "$triple")
  slice="macos-$arch"
  dest="$here/CVDBLite.xcframework/$slice"
  mkdir -p "$dest"
  cp "$a" "$dest/libvdblite.a"
  cp -R "$work/headers" "$dest/Headers"
  cat > "$here/CVDBLite.xcframework/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>AvailableLibraries</key>
  <array>
    <dict>
      <key>HeadersPath</key><string>Headers</string>
      <key>LibraryIdentifier</key><string>$slice</string>
      <key>LibraryPath</key><string>libvdblite.a</string>
      <key>SupportedArchitectures</key><array><string>$arch</string></array>
      <key>SupportedPlatform</key><string>macos</string>
    </dict>
  </array>
  <key>CFBundlePackageType</key><string>XFWK</string>
  <key>XCFrameworkFormatVersion</key><string>1.0</string>
</dict>
</plist>
PLIST
else
  macos="$work/macos/libvdblite.a" sim="$work/ios-sim/libvdblite.a"
  mkdir -p "$work/macos" "$work/ios-sim"
  lipo -create "$(lib aarch64-apple-darwin)" "$(lib x86_64-apple-darwin)" -output "$macos"
  ios=$(lib aarch64-apple-ios)
  lipo -create "$(lib aarch64-apple-ios-sim)" "$(lib x86_64-apple-ios)" -output "$sim"
  xcodebuild -create-xcframework \
    -library "$macos" -headers "$work/headers" \
    -library "$ios" -headers "$work/headers" \
    -library "$sim" -headers "$work/headers" \
    -output "$here/CVDBLite.xcframework"
fi
echo "built $here/CVDBLite.xcframework"

if [ -n "$zip" ]; then
  (cd "$here" && rm -f CVDBLite.xcframework.zip && zip -qry CVDBLite.xcframework.zip CVDBLite.xcframework)
  echo "checksum: $(cd "$here" && swift package compute-checksum CVDBLite.xcframework.zip)"
fi
