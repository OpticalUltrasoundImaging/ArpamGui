**Labeling**
[x] Save to json
[x] Update between frames (save a sequence of annotation lists. One list per frame)
[] When moving annotations with cursor drag, update the model
[x] Connect model update signals to canvas handlers to update the graphics items
[] Model and JSON is a little finicky

**Processing**
[x] Impl SAFT
[x] Impl FWHM in AScan display
[x] Add beamformed RF data to BScan data structure
[] Impl arrival time normalization (flattening)
[] Draw ROI, compute FWHM inside
[] Add vectorization report for clang/MSVC, try to vectorize lib

**Done**
[x] Hotkeys to support selecting frame tools (Default [D], Line [L], Rect [R], etc.)
[x] Hotkey [Ctrl-F] for fullscreen
[x] Float/double proc pipeline
[x] AScan plot

- Float ~150ms
- Double ~160ms
