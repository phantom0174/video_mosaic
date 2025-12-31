#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <omp.h>

#include "common.h"

namespace fs = std::filesystem;

struct ProcessedTile {
    std::vector<float> features;
    float sq_sum;
    std::vector<unsigned char> pixels;
    bool valid = false;
};

int main() {
    if (fs::exists(Cfg::CACHE_FILE)) {
        FILE* fp = fopen(Cfg::CACHE_FILE.c_str(), "rb");
        if (fp) {
            struct Header { int n, d, s; unsigned int m; } h;
            if (fread(&h, sizeof(h), 1, fp) == 1) {
                if (h.m == Cfg::MAGIC && h.d == Cfg::PIXEL_DIM && h.s == Cfg::PLOT_SIZE) {
                    std::cout << ">>> Cache file '" << Cfg::CACHE_FILE << "' found and valid." << std::endl;
                    std::cout << ">>> Skipping generation." << std::endl;
                    std::cout << ">>> Tiles in cache: " << h.n << std::endl;
                    fclose(fp);
                    return 0;
                } else {
                    std::cout << ">>> Cache found but parameters mismatch (Stale). Regenerating..." << std::endl;
                }
            }
            fclose(fp);
        }
    }

    std::cout << ">>> Generating Mosaic Cache <<<" << std::endl;
    std::cout << "Source Dir: " << Cfg::TILE_DIR << std::endl;

    std::vector<std::string> paths;
    if (!fs::exists(Cfg::TILE_DIR)) {
        std::cerr << "Error: Tile directory not found!" << std::endl;
        return 1;
    }

    for (const auto& entry : fs::directory_iterator(Cfg::TILE_DIR)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".JPG") {
                paths.push_back(entry.path().string());
            }
        }
    }
    std::cout << "Found " << paths.size() << " images. Processing..." << std::endl;

    std::vector<ProcessedTile> results(paths.size());
    int processed_count = 0;

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < paths.size(); ++i) {
        int w, h, c;
        unsigned char* img = stbi_load(paths[i].c_str(), &w, &h, &c, 3);
        if (img) {
            results[i].features.resize(Cfg::PIXEL_DIM);
            results[i].pixels.resize(Cfg::PLOT_SIZE);

            std::vector<unsigned char> thumb_u8(Cfg::THUMB_W * Cfg::THUMB_H * 3);
            stbir_resize_uint8_linear(img, w, h, 0, thumb_u8.data(), Cfg::THUMB_W, Cfg::THUMB_H, 0, (stbir_pixel_layout)3);

            float sq_sum = 0.0f;
            for (int k = 0; k < Cfg::PIXEL_DIM; ++k) {
                float val = (float)thumb_u8[k];
                results[i].features[k] = val;
                sq_sum += val * val;
            }
            results[i].sq_sum = sq_sum;

            stbir_resize_uint8_linear(img, w, h, 0, results[i].pixels.data(), Cfg::PLOT_W, Cfg::PLOT_H, 0, (stbir_pixel_layout)3);
            
            results[i].valid = true;
            stbi_image_free(img);
        }

        #pragma omp critical
        {
            processed_count++;
            if (processed_count % 500 == 0) std::cout << "Processed: " << processed_count << "\r" << std::flush;
        }
    }
    std::cout << "\nFinished processing." << std::endl;

    std::vector<float> final_features;
    std::vector<float> final_sq_sums;
    std::vector<unsigned char> final_pixels;
    int valid_count = 0;

    for (const auto& tile : results) {
        if (tile.valid) {
            final_features.insert(final_features.end(), tile.features.begin(), tile.features.end());
            final_sq_sums.push_back(tile.sq_sum);
            final_pixels.insert(final_pixels.end(), tile.pixels.begin(), tile.pixels.end());
            valid_count++;
        }
    }

    FILE* fp = fopen(Cfg::CACHE_FILE.c_str(), "wb");
    if (!fp) {
        std::cerr << "Error: Cannot write to " << Cfg::CACHE_FILE << std::endl;
        return 1;
    }

    struct Header { int n, d, s; unsigned int m; } h;
    h.n = valid_count;
    h.d = Cfg::PIXEL_DIM;
    h.s = Cfg::PLOT_SIZE;
    h.m = Cfg::MAGIC;

    fwrite(&h, sizeof(h), 1, fp);
    fwrite(final_features.data(), sizeof(float), final_features.size(), fp);
    fwrite(final_sq_sums.data(), sizeof(float), final_sq_sums.size(), fp);
    fwrite(final_pixels.data(), sizeof(unsigned char), final_pixels.size(), fp);
    fclose(fp);

    std::cout << "Successfully saved " << valid_count << " tiles to " << Cfg::CACHE_FILE << std::endl;
    return 0;
}