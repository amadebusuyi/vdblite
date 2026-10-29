#!/usr/bin/env bash
# Builds the C library (debug) and runs the C and C++ tests against it.
# Usage: bindings/c/scripts/test.sh
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$here/../.." && pwd)
cd "$root"

# Every function in the header is exported, and every export is declared.
header=$(grep -oE '\bvdb_[a-z_]+\(' "$here/include/vdblite.h" | tr -d '(' | sort -u)
exports=$(grep -oE 'extern "C" fn vdb_[a-z_]+' "$here/src/lib.rs" | sed 's/.* //' | sort -u)
if [ "$header" != "$exports" ]; then
  echo "vdblite.h and src/lib.rs disagree:" >&2
  diff <(echo "$header") <(echo "$exports") >&2 || true
  exit 1
fi

cargo build -p vdblite-c
lib="$root/target/debug"
tmp=$(mktemp -d)
cc="${CC:-cc}"
cxx="${CXX:-c++}"
case "$(uname -s)" in
  Darwin) sys="-liconv -lm" ;;
  *) sys="-lpthread -ldl -lm" ;;
esac
# Link the static library, as Go and Swift do; the C++ test uses the
# shared one, as Java and .NET do.
$cc -std=c11 -Wall -Wextra -Werror -I"$here/include" "$here/tests/test.c" \
  "$lib/libvdblite.a" $sys -o "$tmp/test_c"
$cxx -std=c++17 -Wall -Wextra -Werror -I"$here/include" "$here/tests/test.cpp" \
  -L"$lib" -lvdblite -Wl,-rpath,"$lib" -o "$tmp/test_cpp"
"$tmp/test_c" "$tmp"
"$tmp/test_cpp" "$tmp"
