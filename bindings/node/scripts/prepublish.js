#!/usr/bin/env node
// Prepares the npm packages for a release, after CI has put each platform's
// addon at npm/<platform>/vdblite.node (see scripts/build.js --target).
//
//   node scripts/prepublish.js
//   node scripts/prepublish.js --partial   only the platforms built so far
//                                          (for trying packages locally)
//
// Writes npm/<platform>/package.json and README.md for every platform, and
// lists those packages as optionalDependencies of vdblite at the same
// version, so `npm install vdblite` fetches only the one matching the
// machine. Fails if any platform's addon is missing, unless --partial.
'use strict'

const fs = require('node:fs')
const path = require('node:path')
const { PLATFORMS: ALL } = require('./platforms')

const pkgDir = path.join(__dirname, '..')
const mainPath = path.join(pkgDir, 'package.json')
const main = JSON.parse(fs.readFileSync(mainPath, 'utf8'))

const built = (p) => fs.existsSync(path.join(pkgDir, 'npm', p.name, 'vdblite.node'))
const partial = process.argv.includes('--partial')
const PLATFORMS = partial ? ALL.filter(built) : ALL
const missing = ALL.filter((p) => !built(p))
if (missing.length && !partial) {
  console.error(`missing prebuilt addons: ${missing.map((p) => p.name).join(', ')}`)
  process.exit(1)
}

for (const p of PLATFORMS) {
  const dir = path.join(pkgDir, 'npm', p.name)
  const pkg = {
    name: `@vdblite/${p.name}`,
    version: main.version,
    description: `Prebuilt vdblite addon for ${p.name}`,
    license: main.license,
    repository: main.repository,
    main: 'vdblite.node',
    files: ['vdblite.node'],
    os: [p.os],
    cpu: [p.cpu],
    ...(p.libc && { libc: [p.libc] }),
    engines: main.engines,
    publishConfig: { access: 'public' },
  }
  fs.writeFileSync(path.join(dir, 'package.json'), `${JSON.stringify(pkg, null, 2)}\n`)
  fs.writeFileSync(
    path.join(dir, 'README.md'),
    `# @vdblite/${p.name}\n\nThe prebuilt native addon of [vdblite](https://www.npmjs.com/package/vdblite) ` +
      `for ${p.name}. Install \`vdblite\` instead; npm picks this package when it matches your machine.\n`,
  )
}

main.optionalDependencies = Object.fromEntries(PLATFORMS.map((p) => [`@vdblite/${p.name}`, main.version]))
fs.writeFileSync(mainPath, `${JSON.stringify(main, null, 2)}\n`)
console.log(`prepared vdblite ${main.version} with ${PLATFORMS.length} platform packages`)
