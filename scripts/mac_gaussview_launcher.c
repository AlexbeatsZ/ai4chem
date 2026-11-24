/* Finder launcher for the locally installed, unmodified Intel GaussView bundle.
 * Build/sign instructions and acceptance scope: docs/design/mac-qc.md. */
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <unistd.h>

int main(int argc, char **argv) {
    (void)argc;
    const char *home = getenv("HOME");
    if (!home) return 1;
    char root[4096], executable[4096], value[16384];
    snprintf(root, sizeof(root), "%s/.local/opt/gaussian16", home);
    setenv("g16root", root, 1);
    snprintf(value, sizeof(value), "%s/g16/bsd:%s/g16", root, root);
    setenv("GAUSS_EXEDIR", value, 1);
    snprintf(value, sizeof(value), "%s/g16/basis", root);
    setenv("G16BASIS", value, 1);
    snprintf(value, sizeof(value), "%s/gv", root);
    setenv("GV_DIR", value, 1);
    snprintf(value, sizeof(value), "%s/Library/Caches/Gaussian16/scratch", home);
    setenv("GAUSS_SCRDIR", value, 1);
    snprintf(value, sizeof(value), "%s/bin:%s/g16:/usr/bin:/bin:/usr/sbin:/sbin", root, root);
    setenv("PATH", value, 1);
    struct rlimit limit;
    if (!getrlimit(RLIMIT_STACK, &limit)) {
        limit.rlim_cur = limit.rlim_max;
        setrlimit(RLIMIT_STACK, &limit);
    }
    snprintf(executable, sizeof(executable), "%s/gv/gview.app/Contents/MacOS/gview", root);
    argv[0] = executable;
    execv(executable, argv);
    perror("GaussView launch");
    return 1;
}
