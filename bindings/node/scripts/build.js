#!/usr/bin/env node
// Builds the native addon and copies it next to index.js as vecfile.node.
'use strict'

const { spawnSync } = require('node:child_process')
const fs = require('node:fs')
const path = require('node:path')

const root = path.resolve(__dirname, '..', '..', '..')
const debug = process.argv.includes('--debug')
const profile = debug ? 'dev' : 'release-node'
const outDir = debug ? 'debug' : 'release-node'

// rustup puts cargo in ~/.cargo/bin, which shells only pick up after a restart.
const home = process.env.HOME || process.env.USERPROFILE || ''
const fallback = path.join(home, '.cargo', 'bin', process.platform === 'win32' ? 'cargo.exe' : 'cargo')
const cargo = spawnSync('cargo', ['--version']).error && fs.existsSync(fallback) ? fallback : 'cargo'

const result = spawnSync(cargo, ['build', '-p', 'vecfile-node', '--profile', profile], {
  cwd: root,
  stdio: 'inherit',
})
if (result.error) {
  console.error(`could not run cargo (is Rust installed? https://rustup.rs): ${result.error.message}`)
  process.exit(1)
}
if (result.status !== 0) process.exit(result.status ?? 1)

const lib = {
  darwin: 'libvecfile_node.dylib',
  linux: 'libvecfile_node.so',
  win32: 'vecfile_node.dll',
}[process.platform]
if (!lib) {
  console.error(`unsupported platform ${process.platform}`)
  process.exit(1)
}
const target = process.env.CARGO_TARGET_DIR || path.join(root, 'target')
const src = path.join(target, outDir, lib)
const dest = path.join(__dirname, '..', 'vecfile.node')
fs.copyFileSync(src, dest)
console.log(`built ${path.relative(process.cwd(), dest)}`)
