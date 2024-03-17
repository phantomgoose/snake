# What's this?

Snake implemented in Rust, using [macroquad](https://macroquad.rs/) to visualize the game state.

Not actually "playable" in the traditional sense. Instead, this project relies on the use of neural network and evolutionary algorithms to train NNs to play Snake.

Run with `cargo run --release` for optimal performance.

Press `Esc` or `Q` to quit, `V` to trigger training.

By default, a thousand snake games run in parallel until they're all dead or the simulation times out (timeout is a few seconds by default). The best 20% of the snakes are then selected to populate the next generation. This process repeats until the desired generation number is reached, and the best snake from the last generation then plays the game for your entertainment.

Training currently runs on the CPU and takes a couple minutes on an M2 Max to reach generation 200, by which time the snakes are generally pretty adept at the game.
