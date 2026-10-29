# Releasing VDBLite

A release is a pushed tag. [release.yml](.github/workflows/release.yml) does
the rest. These steps are needed once, before the first release.

## One-time setup

1. **Repository owner.** Replace the `OWNER` placeholder with the real GitHub
   owner, spelled exactly as on GitHub. It appears in repository URLs, the Go
   module path (`github.com/OWNER/vdblite/bindings/go`), the Maven group
   (`io.github.OWNER`) and the READMEs:

   ```sh
   grep -rlI --exclude-dir={.git,target,node_modules,data,.build} --exclude=RELEASING.md \
     -e OWNER . | grep -v scripts/release/ | xargs sed -i '' 's/OWNER/<owner>/g'
   ```

   npm provenance checks that `repository.url` matches the repository that
   publishes, and `version.mjs check` refuses to release while `OWNER` remains
   in a manifest.

2. **Public repository.** The release builds and tests on `ubuntu-24.04-arm`
   and `windows-11-arm`, which are free for public repositories only.

3. **npm.**
   - Create the npm organization `vdblite` (free for public packages) at
     https://www.npmjs.com/org/create, so the `@vdblite/<platform>` packages
     can be published.
   - Create a granular access token with publish rights to `vdblite` and the
     `@vdblite` scope, and add it to the repository as the `NPM_TOKEN` secret.
   - After the first release, configure **trusted publishing** for each of
     the 9 packages (`vdblite` and the 8 `@vdblite/*`) on npmjs.com:
     repository `<owner>/vdblite`, workflow `release.yml`, environment
     `release`. Then delete the `NPM_TOKEN` secret. The workflow uses trusted
     publishing whenever the secret is absent.

4. **crates.io.**
   - Create an API token with the `publish-new` and `publish-update` scopes
     and add it as the `CARGO_REGISTRY_TOKEN` secret.
   - After the first release, add a **trusted publisher** to each of the five
     crates (`vdblite`, `vdblite-core`, `vdblite-index`, `vdblite-chunk`,
     `vdblite-redb`):
     repository `<owner>/vdblite`, workflow `release.yml`, environment
     `release`. Then delete the `CARGO_REGISTRY_TOKEN` secret.

5. **Maven Central** (Java).
   - Sign in at https://central.sonatype.com with the GitHub account: the
     namespace `io.github.<owner>` is verified for you.
   - Generate a user token (Account → Generate User Token) and add its two
     halves as the `MAVEN_CENTRAL_USERNAME` and `MAVEN_CENTRAL_TOKEN` secrets.
   - Central requires signed artifacts. Create a GPG key, publish it
     (`gpg --keyserver keyserver.ubuntu.com --send-keys <id>`), and add the
     armored private key (`gpg --armor --export-secret-keys <id>`) and its
     passphrase as `MAVEN_GPG_PRIVATE_KEY` and `MAVEN_GPG_PASSPHRASE`.

6. **NuGet** (.NET).
   - Create an API key on nuget.org that can push new packages matching
     `VDBLite`, and add it as the `NUGET_API_KEY` secret.
   - After the first release, add a **trusted publishing** policy on
     nuget.org (repository `<owner>/vdblite`, workflow `release.yml`,
     environment `release`), set the repository variable `NUGET_USER` to your
     nuget.org user name, and delete the `NUGET_API_KEY` secret.

7. **Swift.** SwiftPM reads `Package.swift` from the root of a repository at
   a version tag, and the file must carry the checksum of the XCFramework it
   downloads, so the Swift package is published to its own repository.
   - Create the public repository `<owner>/vdblite-swift` with a README, so
     it has a first commit. (Another name: set the repository variable
     `SWIFT_REPO` to `<owner>/<name>`.)
   - Create an SSH key pair (`ssh-keygen -t ed25519 -f swift-deploy -N ""`),
     add the public half to `vdblite-swift` as a deploy key **with write
     access**, and the private half to this repository as the
     `SWIFT_DEPLOY_KEY` secret.

8. **Go** needs no account: the module is fetched from this repository by
   its tag (`bindings/go/vX.Y.Z`), which the release pushes.

9. **GitHub environment.** In Settings → Environments, create `release` and
   add yourself as a required reviewer. Publishing then waits for your
   approval after every build and test has passed.

10. **Private vulnerability reporting.** Turn it on in Settings → Security, as
    [SECURITY.md](SECURITY.md) tells reporters to use it.

## The redb fork

`crates/vdblite-redb` is redb 4.4.0 plus one method (see its FORK.md),
published as `vdblite-redb` and used as `redb`. It is outside the workspace
and versioned on its own: when it changes, give it a new version (for example
`4.4.0-vdblite.1` or the next upstream version), update the `=` pin on `redb`
in `Cargo.toml` to match (`version.mjs check` enforces this), and the next
release publishes it before `vdblite-core`. Until a fork version is on
crates.io, CI's publish dry run skips `vdblite-core` and `vdblite`.

## Each release

```sh
node scripts/release/version.mjs bump 0.2.0
# write the "## [0.2.0]" section in CHANGELOG.md
git commit -am "Release 0.2.0"
git tag v0.2.0
git push --follow-tags
```

The workflow then:

1. checks that the tag, every version number (Cargo, npm, Maven, NuGet) and
   CHANGELOG.md agree;
2. runs the full CI suite;
3. builds the Node addon for 8 platforms and the C library for 9 (the same
   8 plus Windows MinGW for Go), and the Swift XCFramework (macOS, iOS,
   iOS simulator);
4. tests each build on its own platform: the Node tests, and the C, C++,
   Go, Java and .NET tests against the release libraries (Alpine in a
   container, macOS x64 under Rosetta);
5. waits for approval on the `release` environment;
6. publishes, skipping any version already published: the crates in
   dependency order (the `vdblite-redb` fork first, at its own version), the
   9 npm packages (platform packages first), the jar with every platform's
   library to Maven Central, the NuGet package with every runtime, and the
   Go module tag `bindings/go/vX.Y.Z` (a commit, on no branch, that adds the
   static libraries to `bindings/go/lib`);
7. creates the GitHub release from the CHANGELOG section, with the C/C++
   archives and `CVDBLite.xcframework.zip` attached;
8. pushes the Swift package, pointing at that XCFramework, to
   `vdblite-swift` and tags it `vX.Y.Z`;
9. installs each package from its registry on clean machines: npm on macOS,
   Linux (glibc and Alpine, including Debian 10) and Windows; crates.io;
   Go, Maven and NuGet on Linux, macOS and Windows; Swift on macOS.

If a step fails after something was published, fix the cause and re-run the
failed jobs. Published versions are skipped. A broken published version
can't be replaced: release a patch version.
