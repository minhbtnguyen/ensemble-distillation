# ensemble-distillation
`What an Ensemble Buys You, and How Much One Small Model Can Keep?`

## Research Questions
An ensemble costs N times the compute of one model. How much of its quality gain survives when you distill it into one small model, and what latency and size?
- Q1: Why an ensemble helps at all?
- Q2: How it comares to a neural net at equal tuning budget?
- Q3: How much a small student keeps of the teacher's gain, against a control trained on true labels?

## Dataset about
- California Housing is a regression dataset built from the 1990 US Census. Each row is one census block group in California, which covers roughly 1,400 people.
    - Size: 20,640 rows, 8 numeric features, no missing values.
    - Target: median house value for the block group, in units of $100k. It is capped at $500k, so some rows are censored.
