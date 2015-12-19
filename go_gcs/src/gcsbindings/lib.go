package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"syscall"
	"math/rand"
	"time"
	"sync"
	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/cloud"
	"google.golang.org/cloud/storage"
    "code.google.com/p/go-uuid/uuid"
)

//#cgo CFLAGS: -fPIC
import "C"

var projectID = "visualdb-1017"

func getContext (keyPath string) context.Context {
	jsonKey, err := ioutil.ReadFile(keyPath)
	if err != nil { panic(err) }
	conf, err := google.JWTConfigFromJSON(jsonKey, storage.ScopeFullControl)
	if err != nil { panic(err ) }
	return cloud.NewContext(projectID, conf.Client(oauth2.NoContext))
}

func newFifo () string {
	rand.Seed(time.Now().UnixNano())
	fifo := fmt.Sprintf("/var/tmp/visualdb-%s", uuid.New())
	err := syscall.Mkfifo(fifo, 0666)
	if err != nil { panic(err) }
	return fifo
}

var wg_map = struct{
        sync.RWMutex
        m map[string]*sync.WaitGroup
}{m: make(map[string]*sync.WaitGroup)}

//export gcs_read
func gcs_read (key string, storageBucket string, path string) []byte {
	ctx := getContext(key)
	fifo := newFifo()

	rc, err := storage.NewReader(ctx, storageBucket, path)
    // Loop three times to retry because storage api does not provide error
    // code
    i := 0
    for {
        if err != nil {
            if i == 3 {
                panic(err)
            }
            fmt.Print(err.Error() + "\n")
            fmt.Printf("Retrying %d time...\n", i)
            time.Sleep(1e9)
    	    rc, err = storage.NewReader(ctx, storageBucket, path)
            i = i + 1
            continue
        }
        break
    }

	contents, err := ioutil.ReadAll(rc)
	if err != nil { panic(err) }

	go func () {
		fd, err := syscall.Open(fifo, syscall.O_WRONLY, 0)
		if err != nil { panic(err) }

		f := os.NewFile(uintptr(fd), "foobar")
		defer f.Close()
		_, err = f.Write(contents)
		if err != nil { panic(err) }
	} ()

	return []byte(fifo)
}

//export gcs_write
func gcs_write (key string, storageBucket string, path string) []byte {
	ctx := getContext(key)
	fifo := newFifo()

    // To make sure the path name is still valid memory
    // If passed in by a C program, it might be on the stack or cleaned up
    // Probably overkill
    x := []byte(string(path));
    s := string(x);
	wc := storage.NewWriter(ctx, storageBucket, s)

    wg_map.Lock()
    _, ok := wg_map.m[s]
    if !ok {
	    wg_map.m[s] = &sync.WaitGroup{}
    }
    wg_map.Unlock()
    wg_map.RLock()
    (*wg_map.m[s]).Add(1)
    wg_map.RUnlock()
	go func () {
		defer func() {
            wg_map.RLock()
            (*wg_map.m[s]).Done()
            wg_map.RUnlock()
        }()
		fd, err := syscall.Open(fifo, syscall.O_RDONLY, 0)
		if err != nil { panic(err) }

		f := os.NewFile(uintptr(fd), "foobar")
		defer f.Close()

		contents, err := ioutil.ReadAll(f)
		if err != nil { panic(err) }

		_, err = wc.Write(contents[:len(contents)])
		if err != nil { panic(err) }

		err = wc.Close()
        i := 0
        for {
            if err != nil {
                if i == 3 {
                    panic(err)
                }
                fmt.Print(err.Error() + "\n")
                fmt.Printf("Retrying %d time...\n", i)
                time.Sleep(1e9)
    	        err = wc.Close()
                i = i + 1
                continue
            }
            break
        }
	} ()

	return []byte(fifo)
}

//export gcs_object_exists
func gcs_object_exists (key string, storageBucket string, path string) bool {
	ctx := getContext(key)

    _, err := storage.StatObject(ctx, storageBucket, string(path))
    if err == storage.ErrObjectNotExist {
        return false
    }
    if err != nil {
        panic(err)
        return false
    }
    return true
}

//export gcs_ensure_writes_are_done
func gcs_ensure_writes_are_done (path string) {
    wg_map.RLock()
    x := wg_map.m[path]
    wg_map.RUnlock()
    (*x).Wait()
    wg_map.Lock()
	delete(wg_map.m, path)
    wg_map.Unlock()
}
func main () {}
