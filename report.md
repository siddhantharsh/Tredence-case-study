# case study: self pruning neural network

## 1. how the pruning works

the whole point of this project was to make a neural network that can cut off its own connections while training. i did this using sigmoid gates and an l1 penalty.

the way l1 works is pretty simple. it adds up the absolute values of whatever you apply it to. the key thing is that the gradient of l1 is always the same size no matter how small the value gets. so it keeps pushing stuff toward zero with the same force the whole time. l2 doesn't do this, l2's gradient gets weaker as the value shrinks, so things never fully reach zero. that's why l1 is better for pruning.

in my `prunablelinear` layer, i have these raw `gate_scores` that get passed through sigmoid so they're always between 0 and 1. the l1 penalty just adds up all these gate values. during training, the penalty keeps telling the optimizer to make the raw scores more and more negative. when a raw score goes very negative, sigmoid squishes it to basically 0, which kills that connection.

### why i made certain choices
1. **starting value for gates:** i set all `gate_scores` to `3.0` at the start. sigmoid(3) gives about 0.95, so all connections are almost fully on when training begins. if i had used 0 instead, sigmoid(0) = 0.5, so the network would start half broken and couldn't learn properly.
2. **using mean instead of sum for l1:** there are about 3.8 million gates total. if you just sum them all up, you get an l1 value in the millions while the cross entropy loss is like 2.3. that makes it really hard to pick good lambda values. taking the mean instead keeps the l1 between 0 and 1, which is way easier to work with.

### lambda warmup schedule

one big problem i ran into early on was that if you turn on the l1 penalty from epoch 1, the network never gets a chance to learn anything useful before the pruning pressure starts killing connections. you end up pruning random weights instead of the ones that actually don't matter.

to fix this i wrote a simple warmup function called `get_current_lambda`. it works in three phases:
- **epochs 1 to 15:** lambda = 0. no pruning at all. the network just learns features normally.
- **epochs 16 to 35:** lambda ramps up linearly from 0 to the target value. this gives the network time to adjust instead of getting hit with full pressure all at once.
- **epochs 36 to 50:** lambda stays at the max value. this is where most of the pruning happens.

the warmup lets us use way higher lambda values (like 2.5) without destroying accuracy, because by the time the pressure kicks in, the network already knows which connections it needs and which ones are dead weight.

## 2. results

trained on cifar 10 for 50 epochs with adam. the first 15 epochs have no pruning pressure, then it ramps up over 20 epochs. tested three different max lambda values.

| $\lambda$ (max sparsity pressure) | test accuracy | sparsity level (%) |
| :--- | :--- | :--- |
| **0.1** (low) | 60.22% | 18.03% |
| **1.0** (medium) | 60.42% | 51.64% |
| **2.5** (high) | 60.88% | 66.31% |

*sparsity level = percentage of gates that dropped below 0.01.*

## 3. what this means

the warmup schedule made a big difference. without it, high lambda values would tank the accuracy because the network was trying to prune before it even knew what features mattered. with the 15 epoch warmup, the network locks in good weights first and then the l1 penalty can safely prune the connections that turned out to be useless.

at lambda = 2.5, the network got rid of **66.31%** of its connections while actually getting the highest accuracy at **60.88%**. that means most of those connections were doing nothing useful and the network runs better without them.