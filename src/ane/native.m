// Private ANE selector sequence informed by maderix/ANE
// and computer-graphics-tools/ane. See THIRD-PARTY-NOTICES.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <dlfcn.h>
#include <errno.h>
#include <sys/stat.h>
#include <mach/mach.h>
#include <time.h>

typedef struct {
    IOSurfaceRef input;
    IOSurfaceRef output;
    size_t inputBytes;
    size_t outputBytes;
    size_t references;
} ProjectionIo;

typedef struct {
    id model;
    id descriptor;
    id request;
    id inputObject;
    id outputObject;
    NSData *mil;
    NSString *directory;
    ProjectionIo *io;
    bool loaded;
    bool loadAttempted;
    bool ownsDirectory;
    uint64_t directoryDevice;
    uint64_t directoryInode;
    double compileSeconds;
    double loadSeconds;
} DenseHandle;

static double monotonic_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0;
    return (double)value.tv_sec + (double)value.tv_nsec / 1e9;
}

int dyn_ane_memory_stats(uint64_t *footprint, uint64_t *resident) {
    struct task_vm_info info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t status = task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count);
    if (status != KERN_SUCCESS) return -1;
    *footprint = info.phys_footprint;
    *resident = info.resident_size;
    return 0;
}

int dyn_ane_compile_stages(void *handle, double *compile, double *load) {
    DenseHandle *h = handle;
    if (!h || !h->loaded) return -1;
    *compile = h->compileSeconds;
    *load = h->loadSeconds;
    return 0;
}

static void set_error(char *buffer, size_t capacity, NSString *message) {
    if (capacity) snprintf(buffer, capacity, "%s", message.UTF8String ?: "unknown ANE error");
}
static void exception_error(char *buffer, size_t capacity, const char *stage, NSException *exception) {
    @try {
        if (capacity) snprintf(buffer, capacity, "%s exception %s: %s", stage,
            exception.name.UTF8String ?: "unknown", exception.reason.UTF8String ?: "unknown");
    } @catch (__unused NSException *reportingException) {
        if (capacity) snprintf(buffer, capacity, "%s: Objective-C exception (description unavailable)", stage);
    }
}
static NSString *failure(NSString *stage, NSError *error) {
    return [NSString stringWithFormat:@"%@: %@", stage, error ?: @"private API returned failure without NSError"];
}
int dyn_ane_dense_identifier_component_valid(const uint8_t *bytes, size_t length) {
    if (!bytes || !length || length > 255) return 0;
    if (bytes[0] == '.' && (length == 1 || (length == 2 && bytes[1] == '.'))) return 0;
    for (size_t i = 0; i < length; i++) {
        uint8_t c = bytes[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-')) return 0;
    }
    return 1;
}
static bool valid_identifier(id identifier) {
    if (![identifier isKindOfClass:[NSString class]]) return false;
    NSString *value = identifier;
    return dyn_ane_dense_identifier_component_valid((const uint8_t *)value.UTF8String,
        [value lengthOfBytesUsingEncoding:NSUTF8StringEncoding]);
}
static NSString *identifier_details(id identifier) {
    bool isString = [identifier isKindOfClass:[NSString class]];
    NSString *type = identifier ? NSStringFromClass([identifier class]) : @"nil";
    NSString *value = isString ? identifier : ([identifier description] ?: @"(nil)");
    NSData *json = [NSJSONSerialization dataWithJSONObject:@[value] options:0 error:nil];
    NSString *escaped = [[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding];
    return [NSString stringWithFormat:@"class=%@ value=%@ length=%lu", type,
        escaped ?: @"[unavailable]", (unsigned long)(isString ? [identifier length] : 0)];
}
static NSData *creation_mil(const uint8_t *mil, size_t length) {
    static const char marker[] = "name=string(\"mm\")";
    const size_t markerLength = sizeof(marker) - 1;
    if (!mil || length < markerLength || length > SIZE_MAX - 33)
        @throw [NSException exceptionWithName:@"ANE" reason:@"MIL matmul name marker is missing or source extent is invalid" userInfo:nil];
    size_t found = SIZE_MAX;
    for (size_t at = 0; at <= length - markerLength; at++) {
        if (memcmp(mil + at, marker, markerLength) != 0) continue;
        if (found != SIZE_MAX)
            @throw [NSException exceptionWithName:@"ANE" reason:@"MIL matmul name marker is duplicated" userInfo:nil];
        found = at;
    }
    if (found == SIZE_MAX)
        @throw [NSException exceptionWithName:@"ANE" reason:@"MIL matmul name marker is missing" userInfo:nil];
    NSString *identity = [[[NSUUID UUID] UUIDString] stringByReplacingOccurrencesOfString:@"-" withString:@""];
    NSData *nonce = [identity dataUsingEncoding:NSASCIIStringEncoding];
    if (nonce.length != 32)
        @throw [NSException exceptionWithName:@"ANE" reason:@"MIL creation identity is not 32 hexadecimal bytes" userInfo:nil];
    const uint8_t *hex = nonce.bytes;
    for (size_t at = 0; at < nonce.length; at++) {
        uint8_t c = hex[at];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')))
            @throw [NSException exceptionWithName:@"ANE" reason:@"MIL creation identity is not hexadecimal" userInfo:nil];
    }
    size_t insertion = found + markerLength - 2;
    NSMutableData *source = [NSMutableData dataWithCapacity:length + 33];
    [source appendBytes:mil length:insertion];
    [source appendBytes:"_" length:1];
    [source appendData:nonce];
    [source appendBytes:mil + insertion length:length - insertion];
    return [source copy];
}
// CPU-only access to the same creation bytes passed to the model descriptor.
int dyn_ane_dense_test_copy_creation_mil(const uint8_t *mil, size_t length,
        uint8_t *output, size_t capacity, size_t *written) {
    @autoreleasepool {
        @try {
            NSData *source = creation_mil(mil, length);
            if (!source || !output || !written || source.length > capacity) return -1;
            memcpy(output, source.bytes, source.length);
            *written = source.length;
            return 0;
        } @catch (__unused NSException *exception) { return -1; }
    }
}
static IOSurfaceRef surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}
// Rust confines all pair and program operations to one serialized owner thread.
static bool retain_io(ProjectionIo *io, char *error, size_t errorCap) {
    if (!io || !io->references || io->references == SIZE_MAX) {
        set_error(error, errorCap, @"invalid shared I/O reference count");
        return false;
    }
    io->references++;
    return true;
}
static void release_io(ProjectionIo *io) {
    if (!io || !io->references) return;
    if (--io->references) return;
    if (io->input) CFRelease(io->input);
    if (io->output) CFRelease(io->output);
    free(io);
}
void *dyn_ane_projection_io_create(size_t inputBytes, size_t outputBytes,
        char *error, size_t errorCap, int *cleanupComplete) {
    @autoreleasepool {
        *cleanupComplete = 1;
        ProjectionIo *io = calloc(1, sizeof(*io));
        if (!io) { set_error(error, errorCap, @"shared I/O allocation failed"); return NULL; }
        io->references = 1;
        @try {
            if (!inputBytes || !outputBytes || inputBytes % 2 || outputBytes % 2)
                @throw [NSException exceptionWithName:@"ANE" reason:@"invalid shared I/O byte extents" userInfo:nil];
            io->inputBytes = inputBytes;
            io->outputBytes = outputBytes;
            io->input = surface(inputBytes);
            io->output = surface(outputBytes);
            if (!io->input || !io->output || IOSurfaceGetAllocSize(io->input) < inputBytes ||
                    IOSurfaceGetAllocSize(io->output) < outputBytes)
                @throw [NSException exceptionWithName:@"ANE" reason:@"IOSurface allocation failed or is undersized" userInfo:nil];
            return io;
        } @catch (NSException *exception) {
            exception_error(error, errorCap, "shared I/O create", exception);
            @try { release_io(io); }
            @catch (__unused NSException *cleanupException) {
                *cleanupComplete = 0;
                fprintf(stderr, "ANE failed-create shared I/O resources retained\n");
            }
            return NULL;
        }
    }
}
int dyn_ane_projection_io_destroy(void *handle, char *error, size_t errorCap) {
    @autoreleasepool {
        @try { release_io(handle); return 0; }
        @catch (NSException *exception) {
            exception_error(error, errorCap, "shared I/O destroy", exception);
            return -1;
        }
    }
}
int dyn_ane_dense_remove_owned_directory(const char *path, uint64_t device, uint64_t inode,
        char *error, size_t errorCap) {
    @autoreleasepool {
        @try {
            struct stat current;
            if (lstat(path, &current) != 0) {
                int status = errno;
                if (status == ENOENT) return 0;
                if (errorCap) snprintf(error, errorCap, "temporary directory inspection failed: %s", strerror(status));
                return -1;
            }
            if (!S_ISDIR(current.st_mode) || (uint64_t)current.st_dev != device ||
                    (uint64_t)current.st_ino != inode) {
                if (errorCap) snprintf(error, errorCap, "temporary directory identity changed; path was not removed");
                return -1;
            }
            NSString *directory = [[NSFileManager defaultManager] stringWithFileSystemRepresentation:path length:strlen(path)];
            NSError *e = nil;
            if ([[NSFileManager defaultManager] removeItemAtPath:directory error:&e]) return 0;
            // Unload may already remove the directory. A missing child alone is not success.
            if (lstat(path, &current) != 0 && errno == ENOENT) return 0;
            set_error(error, errorCap, failure(@"temporary directory removal", e));
            return -1;
        } @catch (NSException *exception) {
            exception_error(error, errorCap, "temporary directory cleanup", exception);
            return -1;
        }
    }
}
static void claim_directory(DenseHandle *h, int *cleanupComplete) {
    if (mkdir(h->directory.fileSystemRepresentation, 0700) != 0)
        @throw [NSException exceptionWithName:@"ANE" reason:[NSString stringWithFormat:@"temporary directory was not created exclusively: %s", strerror(errno)] userInfo:nil];
    struct stat created;
    if (lstat(h->directory.fileSystemRepresentation, &created) != 0 || !S_ISDIR(created.st_mode)) {
        *cleanupComplete = 0;
        @throw [NSException exceptionWithName:@"ANE" reason:@"created temporary directory could not be identified; path was retained" userInfo:nil];
    }
    h->directoryDevice = (uint64_t)created.st_dev;
    h->directoryInode = (uint64_t)created.st_ino;
    h->ownsDirectory = true;
}
static bool release_handle(DenseHandle *h, char *error, size_t errorCap) {
    h->request = nil;
    bool removed = !h->ownsDirectory || dyn_ane_dense_remove_owned_directory(
        h->directory.fileSystemRepresentation, h->directoryDevice, h->directoryInode, error, errorCap) == 0;
    h->model = nil;
    h->descriptor = nil;
    h->mil = nil;
    h->directory = nil;
    h->inputObject = nil;
    h->outputObject = nil;
    release_io(h->io);
    h->io = NULL;
    free(h);
    return removed;
}
static int finish_destroy(DenseHandle *h, bool unloaded, char *error, size_t errorCap) {
    if (!unloaded) return -1;
    return release_handle(h, error, errorCap) ? 0 : -2;
}

// CPU-only failed-create cleanup: no IOSurfaces, ANE classes or selectors.
int dyn_ane_dense_test_claim_and_cleanup(const char *path, char *error, size_t errorCap,
        int *cleanupComplete) {
    @autoreleasepool {
        *cleanupComplete = 1;
        DenseHandle *h = calloc(1, sizeof(*h));
        if (!h) { set_error(error, errorCap, @"handle allocation failed"); return -1; }
        int result = 0;
        @try {
            h->directory = [[NSFileManager defaultManager] stringWithFileSystemRepresentation:path length:strlen(path)];
            claim_directory(h, cleanupComplete);
        } @catch (NSException *exception) {
            exception_error(error, errorCap, "claim", exception);
            result = -1;
        }
        @try {
            char cleanup[2048] = {0};
            if (!release_handle(h, cleanup, sizeof(cleanup))) {
                *cleanupComplete = 0;
                if (errorCap) snprintf(error, errorCap, "claimed-directory cleanup failed: %s", cleanup);
                return -2;
            }
        } @catch (NSException *exception) {
            *cleanupComplete = 0;
            exception_error(error, errorCap, "claim cleanup", exception);
            return -2;
        }
        return result;
    }
}

int dyn_ane_dense_test_distinct_live_directories(void *first, void *second) {
    @autoreleasepool {
        @try {
            DenseHandle *a = first, *b = second;
            if (!a || !b || !a->loaded || !b->loaded || !a->ownsDirectory || !b->ownsDirectory ||
                    [a->directory isEqualToString:b->directory]) return -1;
            DenseHandle *handles[] = {a, b};
            for (size_t index = 0; index < 2; index++) {
                DenseHandle *h = handles[index];
                struct stat current;
                if (lstat(h->directory.fileSystemRepresentation, &current) != 0 ||
                        !S_ISDIR(current.st_mode) || (uint64_t)current.st_dev != h->directoryDevice ||
                        (uint64_t)current.st_ino != h->directoryInode) return -1;
            }
            return 0;
        } @catch (__unused NSException *exception) { return -1; }
    }
}

// CPU-only lifecycle test: no IOSurface allocation, framework loading or private selectors.
int dyn_ane_dense_test_shared_io_lifetime(void) {
    ProjectionIo *io = calloc(1, sizeof(*io));
    DenseHandle *first = calloc(1, sizeof(*first));
    DenseHandle *second = calloc(1, sizeof(*second));
    DenseHandle *failedCreate = calloc(1, sizeof(*failedCreate));
    if (!io || !first || !second || !failedCreate) {
        free(io); free(first); free(second); free(failedCreate); return 1;
    }
    io->references = 1;
    if (!retain_io(io, NULL, 0)) return 2;
    first->io = io;
    if (!retain_io(io, NULL, 0)) return 3;
    second->io = io;
    if (!retain_io(io, NULL, 0)) return 4;
    failedCreate->io = io;
    if (io->references != 4 || !release_handle(failedCreate, NULL, 0) || io->references != 3) return 5;
    release_io(io); // Drop the public pair owner while programs still retain it.
    if (io->references != 2 || finish_destroy(first, true, NULL, 0) != 0 || io->references != 1) return 6;
    if (finish_destroy(second, false, NULL, 0) != -1 || io->references != 1) return 7;
    // The retained failed-unload handle keeps the final pair reference alive.
    return finish_destroy(second, true, NULL, 0) == 0 ? 0 : 8;
}
static bool unload(DenseHandle *h, char *error, size_t errorCap) {
    if (!h->loadAttempted) return true;
    NSError *e = nil;
    if (!((BOOL(*)(id,SEL,unsigned int,NSError **))objc_msgSend)(h->model,
            @selector(unloadWithQoS:error:), 21, &e)) {
        set_error(error, errorCap, failure(@"unload", e));
        return false;
    }
    h->loaded = false;
    h->loadAttempted = false;
    return true;
}

void *dyn_ane_dense_create(const uint8_t *mil, size_t milLen,
        void *ioHandle, size_t inputBytes,
        size_t outputBytes, char *error, size_t errorCap, int *cleanupComplete) {
    @autoreleasepool {
        *cleanupComplete = 1;
        DenseHandle *h = calloc(1, sizeof(*h));
        if (!h) { set_error(error, errorCap, @"handle allocation failed"); return NULL; }
        @try {
            ProjectionIo *io = ioHandle;
            if (!io || io->inputBytes != inputBytes || io->outputBytes != outputBytes ||
                    !retain_io(io, error, errorCap)) {
                set_error(error, errorCap, @"program and shared I/O extents differ or owner is invalid");
                release_handle(h, NULL, 0); return NULL;
            }
            h->io = io;
            if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)) {
                set_error(error, errorCap, @"AppleNeuralEngine framework could not be loaded");
                release_handle(h, NULL, 0); return NULL;
            }
            Class descClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
            Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
            Class requestClass = NSClassFromString(@"_ANERequest");
            Class ioClass = NSClassFromString(@"_ANEIOSurfaceObject");
            if (!descClass || !modelClass || !requestClass || !ioClass) {
                set_error(error, errorCap, @"required private ANE classes are absent");
                release_handle(h, NULL, 0); return NULL;
            }
            h->mil = creation_mil(mil, milLen);
            h->descriptor = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(descClass,
                @selector(modelWithMILText:weights:optionsPlist:), h->mil, @{}, nil);
            if (!h->descriptor) @throw [NSException exceptionWithName:@"ANE" reason:@"model descriptor creation returned nil" userInfo:nil];
            h->model = ((id(*)(Class,SEL,id))objc_msgSend)(modelClass,
                @selector(inMemoryModelWithDescriptor:), h->descriptor);
            if (!h->model) @throw [NSException exceptionWithName:@"ANE" reason:@"model creation returned nil" userInfo:nil];
            id identifier = ((id(*)(id,SEL))objc_msgSend)(h->model, @selector(hexStringIdentifier));
            if (!valid_identifier(identifier))
                @throw [NSException exceptionWithName:@"ANE"
                    reason:[NSString stringWithFormat:@"invalid model temporary-directory identifier: %@; expected one 1..255-byte ASCII component containing only letters, digits, dot, underscore or hyphen, excluding dot and dot-dot", identifier_details(identifier)] userInfo:nil];
            h->directory = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
            claim_directory(h, cleanupComplete);
            NSError *e = nil;
            if (![h->mil writeToFile:[h->directory stringByAppendingPathComponent:@"model.mil"] options:NSDataWritingAtomic error:&e])
                @throw [NSException exceptionWithName:@"ANE" reason:failure(@"temporary model write", e) userInfo:nil];
            double compileStart = monotonic_seconds();
            if (!((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(h->model,
                    @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
                @throw [NSException exceptionWithName:@"ANE" reason:failure(@"compile", e) userInfo:nil];
            }
            h->compileSeconds = monotonic_seconds() - compileStart;
            h->loadAttempted = true;
            double loadStart = monotonic_seconds();
            if (!((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(h->model,
                    @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
                @throw [NSException exceptionWithName:@"ANE" reason:failure(@"load", e) userInfo:nil];
            }
            h->loadSeconds = monotonic_seconds() - loadStart;
            h->loaded = true;
            h->inputObject = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(ioClass, @selector(objectWithIOSurface:), io->input);
            h->outputObject = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(ioClass, @selector(objectWithIOSurface:), io->output);
            if (!h->inputObject || !h->outputObject)
                @throw [NSException exceptionWithName:@"ANE" reason:@"IOSurface wrapping returned nil" userInfo:nil];
            h->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(requestClass,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[h->inputObject], @[@0], @[h->outputObject], @[@0], nil, nil, @0);
            if (!h->request) @throw [NSException exceptionWithName:@"ANE" reason:@"request creation returned nil" userInfo:nil];
            return h;
        } @catch (NSException *exception) {
            exception_error(error, errorCap, "create", exception);
            char cleanup[2048] = {0};
            @try {
                if (unload(h, cleanup, sizeof(cleanup))) {
                    if (!release_handle(h, cleanup, sizeof(cleanup))) {
                        *cleanupComplete = 0;
                        fprintf(stderr, "ANE failed-create directory cleanup failed: %s\n", cleanup);
                    }
                } else {
                    *cleanupComplete = 0;
                    fprintf(stderr, "ANE failed-create resources retained: %s\n", cleanup);
                }
            } @catch (NSException *cleanupException) {
                *cleanupComplete = 0;
                exception_error(cleanup, sizeof(cleanup), "create cleanup", cleanupException);
                fprintf(stderr, "ANE failed-create remaining resources retained: %s\n", cleanup);
            }
            return NULL;
        }
    }
}

int dyn_ane_dense_write(void *handle, const uint16_t *data, size_t elements, char *error, size_t errorCap) {
    @autoreleasepool {
        @try {
            DenseHandle *h = handle;
            if (!h || !h->io || !data || elements != h->io->inputBytes / 2) { set_error(error, errorCap, @"input capacity mismatch"); return -1; }
            ProjectionIo *io = h->io;
            kern_return_t status = IOSurfaceLock(io->input, 0, NULL);
            if (status) { set_error(error, errorCap, [NSString stringWithFormat:@"input IOSurfaceLock failed: %d", status]); return -1; }
            void *base = IOSurfaceGetBaseAddress(io->input);
            if (!base) { IOSurfaceUnlock(io->input, 0, NULL); set_error(error, errorCap, @"input IOSurface has no base address"); return -1; }
            memcpy(base, data, io->inputBytes);
            status = IOSurfaceUnlock(io->input, 0, NULL);
            if (status) { set_error(error, errorCap, [NSString stringWithFormat:@"input IOSurfaceUnlock failed: %d", status]); return -1; }
            return 0;
        } @catch (NSException *exception) {
            exception_error(error, errorCap, "write", exception); return -1;
        }
    }
}
int dyn_ane_dense_evaluate(void *handle, char *error, size_t errorCap) {
    @autoreleasepool {
        DenseHandle *h = handle;
        @try {
            if (!h || !h->loaded || !h->request || !h->io) { set_error(error, errorCap, @"projection is not loaded and bound"); return -1; }
            NSError *e = nil;
            if (!((BOOL(*)(id,SEL,unsigned int,id,id,NSError **))objc_msgSend)(h->model,
                    @selector(evaluateWithQoS:options:request:error:), 21, @{}, h->request, &e)) {
                set_error(error, errorCap, failure(@"evaluate", e)); return -1;
            }
            return 0;
        } @catch (NSException *exception) {
            exception_error(error, errorCap, "evaluate", exception); return -1;
        }
    }
}
int dyn_ane_dense_read(void *handle, uint16_t *data, size_t elements, char *error, size_t errorCap) {
    @autoreleasepool {
        @try {
            DenseHandle *h = handle;
            if (!h || !h->io || !data || elements != h->io->outputBytes / 2) { set_error(error, errorCap, @"output capacity mismatch"); return -1; }
            ProjectionIo *io = h->io;
            kern_return_t status = IOSurfaceLock(io->output, kIOSurfaceLockReadOnly, NULL);
            if (status) { set_error(error, errorCap, [NSString stringWithFormat:@"output IOSurfaceLock failed: %d", status]); return -1; }
            void *base = IOSurfaceGetBaseAddress(io->output);
            if (!base) { IOSurfaceUnlock(io->output, kIOSurfaceLockReadOnly, NULL); set_error(error, errorCap, @"output IOSurface has no base address"); return -1; }
            memcpy(data, base, io->outputBytes);
            status = IOSurfaceUnlock(io->output, kIOSurfaceLockReadOnly, NULL);
            if (status) { set_error(error, errorCap, [NSString stringWithFormat:@"output IOSurfaceUnlock failed: %d", status]); return -1; }
            return 0;
        } @catch (NSException *exception) {
            exception_error(error, errorCap, "read", exception); return -1;
        }
    }
}
int dyn_ane_dense_destroy(void *handle, char *error, size_t errorCap) {
    @autoreleasepool {
        DenseHandle *h = handle;
        if (!h) return 0;
        @try {
            return finish_destroy(h, unload(h, error, errorCap), error, errorCap);
        } @catch (NSException *exception) {
            exception_error(error, errorCap, "destroy", exception); return -1;
        }
    }
}
