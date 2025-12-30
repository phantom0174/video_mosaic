#include <string>

namespace Cfg {
    // ==========================================
    const std::string FIT_DIR = "./bad_frames";
    const std::string TILE_DIR = "./bad_frames";
    const std::string OUTPUT_DIR = "./fit_result";
    
    const std::string CACHE_FILE = "tiles_cache.bin"; 

    // ==========================================
    
    // bad apple frames dimensions: 480x360
    constexpr int BASE_W = 480;
    constexpr int BASE_H = 360;

    constexpr int GRID_COLS = 40;
    constexpr int GRID_ROWS = 40;

    constexpr float OUTPUT_SCALE = 3.0f; 

    // ------------------------------------------ do not modify
    
    constexpr int W = BASE_W; 
    constexpr int H = BASE_H;

    constexpr int W_SIZE = GRID_COLS;
    constexpr int H_SIZE = GRID_ROWS;
    
    constexpr int THUMB_W = W / W_SIZE; 
    constexpr int THUMB_H = H / H_SIZE; 
    constexpr int PIXEL_DIM = THUMB_W * THUMB_H * 3;

    constexpr int PLOT_W = static_cast<int>(THUMB_W * OUTPUT_SCALE);
    constexpr int PLOT_H = static_cast<int>(THUMB_H * OUTPUT_SCALE);
    constexpr int PLOT_SIZE = PLOT_W * PLOT_H * 3;

    constexpr int FINAL_W = W_SIZE * PLOT_W;
    constexpr int FINAL_H = H_SIZE * PLOT_H;
    constexpr int NUM_BLOCKS = W_SIZE * H_SIZE;

    constexpr int THREADS = 16;
    constexpr int BUFFERS = 24;
    constexpr unsigned int MAGIC = 0xCAFEBABE;
}