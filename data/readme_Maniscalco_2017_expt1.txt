Contributor
-----------
Brian Maniscalco
bmaniscalco@gmail.com


Citation
--------
Maniscalco, B., McCurdy, L. Y., Odegaard, B., & Lau, H. (2017). Limited Cognitive Resources Explain a Trade-Off between Perceptual and Metacognitive Vigilance. The Journal of Neuroscience, 37(5), 1213–1224. https://doi.org/10.1523/JNEUROSCI.2271-13.2016

Experiment 1


Experiment details
------------------

(for full details, see the paper cited above)

Circular patches of white noise were presented to the left and right of fixation for 33 ms. A sinusoidal grating was embedded in either the left or right patch of noise. The subject's task was to indicate which patch contained the grating, left or right, with a keypress.

After entering the stimulus discrimination response, subjects rated confidence in the accuracy of their response on a scale of 1 to 4, with 4 being highest. No absolute meaning was attributed to the numbers on the scale, but rather, they indicated relative levels of confidence for stimulus judgments made in this particular experiment. Thus, subjects were encouraged to use all parts of the confidence scale at least some of the time. 

If subjects did not enter response or confidence within 5 seconds of stimulus offset, the next trial commenced automatically. However, if subjects entered both response and confidence, then the next trial was initiated shortly after entry of confidence. In this way, the pacing of the trials depended on subject RT.

Subjects first performed 2 practice blocks of 28 trials each. Subjects received performance feedback after each trial during practice (high pitched tone following correct responses, low pitched tone for incorrect or missing responses). No performance feedback was provided for the remainder of the experiment. 

Subjects then performed a calibration block of 120 trials in order to determine the grating contrast to be used in the main experiment.

In the main experiment, subjects performed 10 blocks of 100 trials each. After each block, there was a self-terminated break period lasting up to 1 minute. There were no experimental manipulations.

The data included here correspond to the main experiment only (i.e. practice and calibration data are not included).

In the data analyzed in the paper (n=26), subjects 3, 6, 9, and 22 were excluded from analysis (subject 3 performed at ceiling, and the others entered an extreme confidence ratings on > 89% of trials, precluding a meaningful calculation of meta-d'). The excluded subjects are included in this data set of n=30.


Data coding
-----------

* Subj_idx
subject number

* Stimulus
0 --> grating was on the left
1 --> grating was on the right

* Response
0 --> subject responded "grating was on the left"
1 --> subject responded "grating was on the right"
NaN --> subject failed to enter response within 5 sec of stimulus offset

* Confidence
1 - 4 --> subject entered confidence 1 - 4
NaN --> subject failed to enter confidence within 5 sec of stimulus offset

* RT_dec
number --> seconds between stimulus onset and entry of response
NaN --> subject failed to enter response within 5 sec of stimulus offset

* RT_conf
number --> seconds between entry of response and entry of confidence
NaN --> subject failed to enter confidence within 5 sec of stimulus offset

* Contrast
Michelson contrast of grating

* BlockNumber
block of trials in which current trial was contained