//go:build !vdblite_custom

package vdblite

// The prebuilt static library for this platform, which release tags of
// this module include. Build with -tags vdblite_custom to link your own
// (set CGO_LDFLAGS).

// #cgo LDFLAGS: ${SRCDIR}/lib/windows_amd64/libvdblite.a -lws2_32 -luserenv -lntdll -lbcrypt -ladvapi32 -lkernel32
import "C"
