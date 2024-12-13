# goal/problem
goal: get good at implementing custom model architectures + doing custom experiments
problem: although I can implement models, they usually have bugs and dont train well

# plan
build to learn. do TONS of reps of implementing/optimizing models. take on ones as challenging as I can handle to get better faster (progressive overload). start=MLPs, goal=custom transformer experiments
  1) implement the model myself, in jax
  2) try to optimize it myself + fix bugs
  3) have chatGPT make its own model to compare against mine
  4) optimize mine til it beats that chatgp's + learn from chatgpt's code

# current subproject (status: DONE)
optimize an CNN to train on MNIST. overfit 1 sample first, then train/val split over all of MNIST.

# progress
: implemented CNN forward pass. it initially trained at ~1 step/min (lol)
: had chatpt implement a cnn for comparison. it got errors. once fixed it trained slowly
: replaced my unoptimized maxpool/conv2d implementations with jax's builtins
  - immediate 200000x speedup. 1 step/min -> 192,805 steps/min. overfits in 2 steps
: got it to successfully get high accuracy on MNIST!! LETS GOOOOOOOOOOOOOOOOOOOOOOOOO
: had chatgpt write pygame code so I could draw digits and have my network predict them
: added a skip connections. why? for fun ig, my whole goal is to run custom experiments sooo
# x -> conv -> relu -> x += conv(x) -> maxpool -> conv -> relu -> x += conv(x) -> fc -> fc
  it trained poorly until I added He initialization + set lr from 0.01 to 0.001. then it did GREAT

# what helped:
- for learning: write functions manually first, then use jax built ins
- for quickly implementing networks: manually add layers, instead of automating their creation
  - for future, err on the side of not prematurely optimizing.dont automate SHIT til ive done it manually first
    - not the first time ive made this mistake on a project lmao and it wont be the last
- for implenting cnns: keep track of the current shape (via a variable + comments) in param init
- id love a shape inspector, so I could hover over a variable and see a 3d image of its shape

## resources (see comments)
[0] cnn architecture visualizer http://alexlenail.me/NN-SVG/AlexNet.html


