This code includes a customer Layer that makes a position zero in the input tensor and passes de gradients along.
Int can be used to error injection if a custom value is added instead on zero as it is a trivial change.
Multiple error can be injected as well but that would not be a realistic HW fault.
This is very usefull to estimate the vulnerability factor of your network. This can be used then in your FMEDA and you might get
friendlier reliability metrics.
