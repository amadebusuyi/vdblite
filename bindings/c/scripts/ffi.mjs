#!/usr/bin/env node
// Builds the C library (bindings/c) and puts it where each language binding
// looks for it.
//
//   node bindings/c/scripts/ffi.mjs dev [--debug]
//       build for this machine and place it for Go, Java and .NET
//   node bindings/c/scripts/ffi.mjs build <platform> [--zig] [--debug]
//       cross-build into target/ffi/<platform>/{include,lib}
//   node bindings/c/scripts/ffi.mjs place <dir>
//       place every <dir>/<platform>/lib for Go, Java and .NET
//   node bindings/c/scripts/ffi.mjs archive <platform> <version> <out-dir>
//       pack target/ffi/<platform> as vdblite-c-<version>-<platform>.tar.gz
//
// Swift has its own script (bindings/swift/scripts/build-xcframework.sh).

import { execFileSync, spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const bindings = path.join(root, 'bindings')

const dylib = { shared: 'libvdblite.dylib', static: 'libvdblite.a' }
const so = { shared: 'libvdblite.so', static: 'libvdblite.a' }
const msvc = { shared: 'vdblite.dll', static: 'vdblite.lib', importLib: 'vdblite.dll.lib' }

// go: GOOS_GOARCH (bindings/go/lib/), jna: JNA's resource prefix, rid: the
// .NET runtime identifier. A platform without one isn't shipped there.
export const PLATFORMS = {
  'darwin-arm64': { target: 'aarch64-apple-darwin', go: 'darwin_arm64', jna: 'darwin-aarch64', rid: 'osx-arm64', ...dylib },
  'darwin-x64': { target: 'x86_64-apple-darwin', go: 'darwin_amd64', jna: 'darwin-x86-64', rid: 'osx-x64', ...dylib },
  'linux-x64-gnu': { target: 'x86_64-unknown-linux-gnu', go: 'linux_amd64', jna: 'linux-x86-64', rid: 'linux-x64', ...so },
  'linux-arm64-gnu': { target: 'aarch64-unknown-linux-gnu', go: 'linux_arm64', jna: 'linux-aarch64', rid: 'linux-arm64', ...so },
  'linux-x64-musl': { target: 'x86_64-unknown-linux-musl', rid: 'linux-musl-x64', ...so },
  'linux-arm64-musl': { target: 'aarch64-unknown-linux-musl', rid: 'linux-musl-arm64', ...so },
  'win32-x64-msvc': { target: 'x86_64-pc-windows-msvc', jna: 'win32-x86-64', rid: 'win-x64', ...msvc },
  'win32-arm64-msvc': { target: 'aarch64-pc-windows-msvc', jna: 'win32-aarch64', rid: 'win-arm64', ...msvc },
  // Go's cgo links with MinGW on Windows.
  'win32-x64-gnu': { target: 'x86_64-pc-windows-gnu', go: 'windows_amd64', shared: 'vdblite.dll', static: 'libvdblite.a' },
}

// The oldest glibc the Linux builds support (as for the Node addon).
const GLIBC = '2.28'

function fail(msg) {
  console.error(msg)
  process.exit(1)
}

function hostPlatform() {
  const host = execFileSync('rustc', ['-vV'], { encoding: 'utf8' }).match(/^host: (.+)$/m)[1]
  const found = Object.entries(PLATFORMS).find(([, p]) => p.target === host)
  if (!found) fail(`no platform for this machine (${host})`)
  return found[0]
}

/** Builds `platform` into target/ffi/<platform>; returns that directory. */
function build(platform, { debug = false, zig = false, host = false } = {}) {
  const p = PLATFORMS[platform] ?? fail(`unknown platform ${platform}; one of ${Object.keys(PLATFORMS).join(', ')}`)
  const profile = debug ? 'dev' : 'release-ffi'
  const args = [zig ? 'zigbuild' : 'build', '-p', 'vdblite-c', '--profile', profile]
  let out = path.join(root, 'target')
  if (!host) {
    const glibc = zig && p.target.endsWith('-gnu') && p.target.includes('linux') ? `.${GLIBC}` : ''
    args.push('--target', p.target + glibc)
    out = path.join(out, p.target)
  }
  out = path.join(out, debug ? 'debug' : 'release-ffi')
  const env = { ...process.env }
  // A musl cdylib must link libc dynamically.
  if (p.target.endsWith('-musl')) env.RUSTFLAGS = `${env.RUSTFLAGS ?? ''} -C target-feature=-crt-static`.trim()
  env.MACOSX_DEPLOYMENT_TARGET ??= '11.0'
  execFileSync('cargo', args, { cwd: root, stdio: 'inherit', env })
  // The system libraries the static library needs, as rustc reports them;
  // this rebuilds only vdblite-c's own crate.
  const rustc = [args[0] === 'build' ? 'rustc' : 'zigbuild', ...(args[0] === 'build' ? [] : ['rustc'])]
  const report = spawnSync('cargo', [...rustc, ...args.slice(1), '--crate-type', 'staticlib', '--', '--print=native-static-libs'], {
    cwd: root,
    env,
    encoding: 'utf8',
  })
  if (report.status !== 0) fail(report.stderr)
  const link = report.stderr.match(/native-static-libs: (.*)$/m)?.[1]?.trim() ?? fail('rustc did not report native-static-libs')

  const dist = path.join(root, 'target', 'ffi', platform)
  fs.rmSync(dist, { recursive: true, force: true })
  fs.mkdirSync(path.join(dist, 'lib'), { recursive: true })
  fs.cpSync(path.join(bindings, 'c', 'include'), path.join(dist, 'include'), { recursive: true })
  for (const file of [p.shared, p.static, p.importLib].filter(Boolean)) {
    fs.copyFileSync(path.join(out, file), path.join(dist, 'lib', file))
  }
  fs.writeFileSync(path.join(dist, 'lib', 'link.txt'), `${link}\n`)
  console.log(`built target/ffi/${platform}`)
  return dist
}

/** Copies `dir`'s libraries to where Go, Java and .NET look for them. */
function place(platform, dir) {
  const p = PLATFORMS[platform]
  const copy = (file, to) => {
    fs.mkdirSync(to, { recursive: true })
    fs.copyFileSync(path.join(dir, 'lib', file), path.join(to, file))
    console.log(`placed ${path.relative(root, path.join(to, file))}`)
  }
  if (p.go) copy(p.static, path.join(bindings, 'go', 'lib', p.go))
  if (p.jna) copy(p.shared, path.join(bindings, 'java', 'src', 'main', 'resources', p.jna))
  if (p.rid) copy(p.shared, path.join(bindings, 'dotnet', 'runtimes', p.rid, 'native'))
}

const [cmd, ...rest] = process.argv.slice(2)
const flag = (name) => rest.includes(name)

if (cmd === 'dev') {
  const platform = hostPlatform()
  place(platform, build(platform, { debug: flag('--debug'), host: true }))
} else if (cmd === 'build') {
  const platform = rest.find((a) => !a.startsWith('--')) ?? fail('usage: ffi.mjs build <platform> [--zig] [--debug]')
  build(platform, { debug: flag('--debug'), zig: flag('--zig') })
} else if (cmd === 'place') {
  const dir = rest[0] ?? fail('usage: ffi.mjs place <dir>')
  for (const platform of fs.readdirSync(dir)) {
    if (PLATFORMS[platform] && fs.existsSync(path.join(dir, platform, 'lib'))) place(platform, path.join(dir, platform))
  }
} else if (cmd === 'archive') {
  const [platform, version, outDir] = rest
  if (!outDir) fail('usage: ffi.mjs archive <platform> <version> <out-dir>')
  const dist = path.join(root, 'target', 'ffi', platform)
  for (const f of ['LICENSE-APACHE']) fs.copyFileSync(path.join(root, f), path.join(dist, f))
  fs.copyFileSync(path.join(bindings, 'c', 'README.md'), path.join(dist, 'README.md'))
  // The archive unpacks into one folder named after it.
  const name = `vdblite-c-${version}-${platform}`
  const stage = path.join(root, 'target', 'ffi-archive')
  fs.rmSync(stage, { recursive: true, force: true })
  fs.cpSync(dist, path.join(stage, name), { recursive: true })
  fs.mkdirSync(outDir, { recursive: true })
  const file = path.resolve(outDir, `${name}.tar.gz`)
  execFileSync('tar', ['-czf', file, name], { cwd: stage, stdio: 'inherit' })
  console.log(`wrote ${path.relative(process.cwd(), file)}`)
} else {
  fail('usage: ffi.mjs dev | build <platform> | place <dir> | archive <platform> <version> <out-dir>')
}
