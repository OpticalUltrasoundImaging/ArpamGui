#pragma once

#include <imgui.h>


namespace arpam_gui {

    class MainGui {
    public:
        MainGui(ImGuiIO& io) : io(io) {}

        void render();

    private:
        ImGuiIO& io;

        // Example states
        bool show_demo_window = true;
        bool show_another_window = false;

    };

}
