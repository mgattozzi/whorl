# whorl - A single file, std only, async Rust executor

whorl was created to teach you how async executors work in Rust. It is not the
fastest executor nor is it's API perfect, but it will teach you about them and
how they work and where to get started if you wanted to make your own. It's
written in a literate programming style such that reading it from beginning to
end tells you a story about how it works or you can read parts of it in chunks
depending on what you want to get out of it.

You can read it all [online here on GitHub](https://github.com/mgattozzi/whorl/blob/main/src/lib.rs)
or you can clone the repo yourself and open up `src/lib.rs` to read through it
in your favorite text editor or play around with it and change things. All of
the code is licensed under the `MIT License` so you're mostly free to do with it
as you wish. If you want to make the next `tokio` or just make something for fun
you can do that.

If you just want to see it in action an example test program is included as part
of the file. You can see it's output by just running:

```bash
cargo test -- --nocapture
```

Which should look something like this:

```bash
whorl on  main [!⇡] is 📦 v0.1.0 via 🦀 v1.56.0 took 10s
❯ cargo test -- --nocapture
   Compiling whorl v0.1.0 (/home/michael/whorl)
    Finished test [unoptimized + debuginfo] target(s) in 0.47s
     Running unittests (target/debug/deps/whorl-6d670ffb5bb225ca)

running 1 test
Begin Asynchronous Execution
Blocking Function Polled To Completion
Spawned Fn #00: Start 1635276666
Spawned Fn #01: Start 1635276666
Spawned Fn #02: Start 1635276666
Spawned Fn #03: Start 1635276666
Spawned Fn #04: Start 1635276666
Spawned Fn #00: Ended 1635276669
Spawned Fn #02: Ended 1635276669
Spawned Fn #03: Ended 1635276669
Spawned Fn #01: Ended 1635276670
Spawned Fn #00: Inner 1635276671
Spawned Fn #03: Inner 1635276674
Spawned Fn #04: Ended 1635276675
Spawned Fn #02: Inner 1635276675
Spawned Fn #01: Inner 1635276678
Spawned Fn #04: Inner 1635276678
End of Asynchronous Execution
test library_test ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 23.00s

   Doc-tests whorl

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

This was originally created for my monthly newsletter. You can find that post in
particular [here](https://mgattozzi.substack.com/p/whorl) or you can sign up for it
[here](https://mgattozzi.substack.com).
