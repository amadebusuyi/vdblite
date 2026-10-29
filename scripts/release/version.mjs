#!/usr/bin/env node
// Keeps every published version number in step.
//
//   node scripts/release/version.mjs check [vX.Y.Z]   all versions agree; with a tag, also
//                                                     that they match it, CHANGELOG.md has
//                                                     its section and no URL says OWNER
//   node scripts/release/version.mjs bump X.Y.Z       set them all, refresh Cargo.lock
//   node scripts/release/version.mjs notes X.Y.Z      print that version's CHANGELOG section
//
// The versions: [workspace.package] version in Cargo.toml, the `=X.Y.Z`
// pins on the internal crates in [workspace.dependencies],
// bindings/node/package.json, the Java pom.xml and the .NET project. The npm
// platform packages take theirs from package.json at publish time
// (bindings/node/scripts/prepublish.js); Go and Swift are versioned by the
// tags the release workflow pushes.

import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
const cargoPath = path.join(root, 'Cargo.toml')
const npmPath = path.join(root, 'bindings', 'node', 'package.json')
// The redb fork keeps its own version; the workspace pins it exactly.
const forkPath = path.join(root, 'crates', 'vdblite-redb', 'Cargo.toml')
const changelogPath = path.join(root, 'CHANGELOG.md')
const pomPath = path.join(root, 'bindings', 'java', 'pom.xml')
const csprojPath = path.join(root, 'bindings', 'dotnet', 'src', 'VDBLite', 'VDBLite.csproj')
// Files whose repository URL or module path names the owner.
const ownerFiles = [
  'Cargo.toml',
  'crates/vdblite-redb/Cargo.toml',
  'bindings/node/package.json',
  'bindings/java/pom.xml',
  'bindings/dotnet/src/VDBLite/VDBLite.csproj',
  'bindings/go/go.mod',
].map((f) => path.join(root, f))
const INTERNAL = ['vdblite-core', 'vdblite-index', 'vdblite-chunk']
const SEMVER = /^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$/

const workspaceVersion = /(\[workspace\.package\][^[]*?\nversion = ")([^"]+)(")/
const internalPin = (name) => new RegExp(`(\\n${name} = \\{[^}]*version = "=)([^"]+)(")`)
// The project's own <version>, the first after its <artifactId>.
const pomVersion = /(<artifactId>vdblite<\/artifactId>\s*(?:<!--[^]*?-->\s*)?<version>)([^<]+)(<\/version>)/
const csprojVersion = /(<Version>)([^<]+)(<\/Version>)/

function versions() {
  const cargo = fs.readFileSync(cargoPath, 'utf8')
  const found = { 'Cargo.toml [workspace.package]': cargo.match(workspaceVersion)?.[2] }
  for (const name of INTERNAL) found[`Cargo.toml ${name} pin`] = cargo.match(internalPin(name))?.[2]
  found['bindings/node/package.json'] = JSON.parse(fs.readFileSync(npmPath, 'utf8')).version
  found['bindings/java/pom.xml'] = fs.readFileSync(pomPath, 'utf8').match(pomVersion)?.[2]
  found['bindings/dotnet VDBLite.csproj'] = fs.readFileSync(csprojPath, 'utf8').match(csprojVersion)?.[2]
  return found
}

function changelogSection(version) {
  const text = fs.readFileSync(changelogPath, 'utf8')
  const start = text.search(new RegExp(`^## \\[${version.replaceAll('.', '\\.')}\\]`, 'm'))
  if (start < 0) return null
  const rest = text.slice(start)
  const next = rest.slice(1).search(/^## \[/m)
  const section = next < 0 ? rest : rest.slice(0, next + 1)
  return section.split('\n').slice(1).join('\n').replace(/\n\[[^\]]+\]: .*$/gm, '').trim()
}

function fail(msg) {
  console.error(msg)
  process.exit(1)
}

const [cmd, arg] = process.argv.slice(2)

if (cmd === 'check') {
  const found = versions()
  const want = arg ? arg.replace(/^v/, '') : found['Cargo.toml [workspace.package]']
  const wrong = Object.entries(found).filter(([, v]) => v !== want)
  for (const [where, v] of Object.entries(found)) console.log(`${v === want ? 'ok  ' : 'BAD '} ${where}: ${v}`)
  if (wrong.length) fail(`versions must all be ${want}${arg ? ` (tag ${arg})` : ''}`)
  console.log(`all versions are ${want}`)
  if (fs.existsSync(forkPath)) {
    const forkVersion = fs.readFileSync(forkPath, 'utf8').match(/^version = "([^"]+)"/m)?.[1]
    const pin = fs.readFileSync(cargoPath, 'utf8').match(/\nredb = \{[^}]*version = "=([^"]+)"/)?.[1]
    if (forkVersion !== pin) fail(`Cargo.toml pins redb (vdblite-redb) at =${pin}, but the fork is ${forkVersion}`)
    console.log(`redb fork pinned at its version ${forkVersion}`)
  }
  // The rest only has to hold for a release.
  if (!arg) process.exit(0)
  if (!changelogSection(want)) fail(`CHANGELOG.md has no "## [${want}]" section`)
  for (const file of ownerFiles.filter((f) => fs.existsSync(f))) {
    if (/\bOWNER\b/.test(fs.readFileSync(file, 'utf8'))) {
      fail(`${path.relative(root, file)} still has the OWNER placeholder`)
    }
  }
  console.log(`ready to release ${arg}`)
} else if (cmd === 'bump') {
  if (!arg || !SEMVER.test(arg)) fail('usage: version.mjs bump X.Y.Z')
  let cargo = fs.readFileSync(cargoPath, 'utf8')
  cargo = cargo.replace(workspaceVersion, `$1${arg}$3`)
  for (const name of INTERNAL) cargo = cargo.replace(internalPin(name), `$1${arg}$3`)
  fs.writeFileSync(cargoPath, cargo)
  const pkg = JSON.parse(fs.readFileSync(npmPath, 'utf8'))
  pkg.version = arg
  fs.writeFileSync(npmPath, `${JSON.stringify(pkg, null, 2)}\n`)
  for (const [file, re] of [[pomPath, pomVersion], [csprojPath, csprojVersion]]) {
    fs.writeFileSync(file, fs.readFileSync(file, 'utf8').replace(re, `$1${arg}$3`))
  }
  const lock = spawnSync('cargo', ['update', '--workspace'], { cwd: root, stdio: 'inherit' })
  if (lock.status !== 0) fail('cargo update --workspace failed; Cargo.lock is stale')
  console.log(`bumped to ${arg}; add a "## [${arg}]" section to CHANGELOG.md, commit, then tag v${arg}`)
} else if (cmd === 'notes') {
  const section = arg && changelogSection(arg.replace(/^v/, ''))
  if (!section) fail(`CHANGELOG.md has no "## [${arg}]" section`)
  console.log(section)
} else {
  fail('usage: version.mjs check [vX.Y.Z] | bump X.Y.Z | notes X.Y.Z')
}
