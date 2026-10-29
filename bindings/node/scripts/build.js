#!/usr/bin/env node
// Builds the native addon.
//
//   node scripts/build.js                     host build -> vdblite.node next to index.js
//   node scripts/build.js --debug             same, unoptimized
//   node scripts/build.js --target <triple>   cross build -> npm/<platform>/vdblite.node
//   node scripts/build.js --target <triple> --zig
//                                             link Linux targets with cargo-zigbuild
//                                             (gnu builds then need only glibc 2.28)
'use strict'

const { spawnSync } = require('node:child_process')
const fs = require('node:fs')
const path = require('node:path')
const { GLIBC, byTriple, libName } = require('./platforms')

const root = path.resolve(__dirname, '..', '..', '..')
const args = process.argv.slice(2)
const debug = args.includes('--debug')
const zig = args.includes('--zig')
const triple = args.includes('--target') ? args[args.indexOf('--target') + 1] : null
const profile = debug ? 'dev' : 'release-node'
const outDir = debug ? 'debug' : 'release-node'

// rustup puts cargo in ~/.cargo/bin, which shells only pick up after a restart.
const home = process.env.HOME || process.env.USERPROFILE || ''
const fallback = path.join(home, '.cargo', 'bin', process.platform === 'win32' ? 'cargo.exe' : 'cargo')
const cargo = spawnSync('cargo', ['--version']).error && fs.existsSync(fallback) ? fallback : 'cargo'

const cargoArgs = [zig ? 'zigbuild' : 'build', '-p', 'vdblite-node', '--profile', profile]
const env = { ...process.env }
let platform = null
if (triple) {
  platform = byTriple(triple)
  // zigbuild picks the glibc version from a suffix on the target name.
  cargoArgs.push('--target', zig && platform.libc === 'glibc' ? `${triple}.${GLIBC}` : triple)
  // A cdylib can't be built against static musl; link the system's libc.so.
  if (platform.libc === 'musl') env.RUSTFLAGS = `${env.RUSTFLAGS ?? ''} -C target-feature=-crt-static`.trim()
}

const result = spawnSync(cargo, cargoArgs, { cwd: root, stdio: 'inherit', env })
if (result.error) {
  console.error(`could not run cargo (is Rust installed? https://rustup.rs): ${result.error.message}`)
  process.exit(1)
}
if (result.status !== 0) process.exit(result.status ?? 1)

const target = process.env.CARGO_TARGET_DIR || path.join(root, 'target')
let src
let dest
if (platform) {
  src = path.join(target, triple, outDir, libName(triple))
  dest = path.join(__dirname, '..', 'npm', platform.name, 'vdblite.node')
} else {
  const lib = {
    darwin: 'libvdblite_node.dylib',
    linux: 'libvdblite_node.so',
    win32: 'vdblite_node.dll',
  }[process.platform]
  if (!lib) {
    console.error(`unsupported platform ${process.platform}`)
    process.exit(1)
  }
  src = path.join(target, outDir, lib)
  dest = path.join(__dirname, '..', 'vdblite.node')
}
fs.mkdirSync(path.dirname(dest), { recursive: true })
fs.copyFileSync(src, dest)
console.log(`built ${path.relative(process.cwd(), dest)}`)
