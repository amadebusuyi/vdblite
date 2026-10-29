//go:build !vdblite_custom

package vdblite

// The prebuilt static library for this platform, which release tags of
// this module include. Build with -tags vdblite_custom to link your own
// (set CGO_LDFLAGS).

// #cgo LDFLAGS: ${SRCDIR}/lib/linux_amd64/libvdblite.a -lpthread -ldl -lm -lrt -lutil
import "C"
