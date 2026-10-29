'use strict'

// The platforms prebuilt addons ship for. Each becomes an npm package
// `@vdblite/<name>` holding one vdblite.node, which npm installs only where
// its os, cpu and libc match. index.js computes the same name at runtime.

const PLATFORMS = [
  { name: 'darwin-arm64', triple: 'aarch64-apple-darwin', os: 'darwin', cpu: 'arm64' },
  { name: 'darwin-x64', triple: 'x86_64-apple-darwin', os: 'darwin', cpu: 'x64' },
  { name: 'linux-x64-gnu', triple: 'x86_64-unknown-linux-gnu', os: 'linux', cpu: 'x64', libc: 'glibc' },
  { name: 'linux-arm64-gnu', triple: 'aarch64-unknown-linux-gnu', os: 'linux', cpu: 'arm64', libc: 'glibc' },
  { name: 'linux-x64-musl', triple: 'x86_64-unknown-linux-musl', os: 'linux', cpu: 'x64', libc: 'musl' },
  { name: 'linux-arm64-musl', triple: 'aarch64-unknown-linux-musl', os: 'linux', cpu: 'arm64', libc: 'musl' },
  { name: 'win32-x64-msvc', triple: 'x86_64-pc-windows-msvc', os: 'win32', cpu: 'x64' },
  { name: 'win32-arm64-msvc', triple: 'aarch64-pc-windows-msvc', os: 'win32', cpu: 'arm64' },
]

// Oldest glibc the gnu builds link against (via cargo-zigbuild): RHEL 8,
// Debian 10, Ubuntu 18.10 and Amazon Linux 2023 or newer.
const GLIBC = '2.28'

function byTriple(triple) {
  const p = PLATFORMS.find((p) => p.triple === triple)
  if (!p) throw new Error(`no prebuilt platform for target ${triple}`)
  return p
}

/** The shared-library file cargo writes for a cdylib on this target. */
function libName(triple) {
  if (triple.includes('apple')) return 'libvdblite_node.dylib'
  if (triple.includes('windows')) return 'vdblite_node.dll'
  return 'libvdblite_node.so'
}

module.exports = { PLATFORMS, GLIBC, byTriple, libName }
