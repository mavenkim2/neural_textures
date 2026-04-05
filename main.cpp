#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "tests/forward_pass_test.h"

int main(int argc, int argv[])
{
    (void)argc;
    (void)argv;

    if (!neural_textures::RunForwardPassTest())
    {
        std::printf("ForwardPassTest failed\n");
        return 1;
    }
}
