Contributor
-----------
Brian Maniscalco
bmaniscalco@gmail.com


Citation
--------
Maniscalco, B., McCurdy, L. Y., Odegaard, B., & Lau, H. (2017). Limited Cognitive Resources Explain a Trade-Off between Perceptual and Metacognitive Vigilance. The Journal of Neuroscience, 37(5), 1213–1224. https://doi.org/10.1523/JNEUROSCI.2271-13.2016

Experiment 3


Experiment details
------------------

(for full details, see the paper cited above)

Circular patches of white noise were presented to the left and right of fixation for 33 ms. A sinusoidal grating was embedded in either the left or right patch of noise. The subject's task was to indicate which patch contained the grating, left or right, with a keypress.

On some (but not all) trials, after entering the stimulus discrimination response, subjects rated confidence in the accuracy of their response on a scale of 1 to 4, with 4 being highest. No absolute meaning was attributed to the numbers on the scale, but rather, they indicated relative levels of confidence for stimulus judgments made in this particular experiment. Thus, subjects were encouraged to use all parts of the confidence scale at least some of the time. 

There was a 2 second period after stimulus offset in which the subject had to enter response and confidence. The full 2 seconds elapsed before the start of the next trial even if response and confidence were entered early. This brief and mandatory response period was introduced in order to standardize block duration for different block types (see below) to facilitate comparison of decision fatigue effects in the different block types.

In odd numbered blocks ("partial type 2 blocks"), subjects did not provide confidence ratings for the first 50 of 100 trials in the block; on even numbered blocks ("whole type 2 blocks"), however, they provided confidence ratings on all trials. (This manipulation was introduced to test a hypothesis about how having to evaluate confidence might influence decision fatigue.) On trials where a confidence rating was collected, text appeared on the screen after entry of the stimulus discrimination response to prompt entry of confidence. Text also appeared on the screen before each block to indicate to the subject whether confidence would be collected on every trial, or only the last 50 trials.

Subjects first performed 2 practice blocks of 28 trials each. Subjects received performance feedback after each trial during practice (high pitched tone following correct responses, low pitched tone for incorrect or missing responses). No performance feedback was provided for the remainder of the experiment. 

Subjects then performed a calibration block of 120 trials in order to determine the grating contrast to be used in the main experiment. 

In the main experiment, subjects performed 10 blocks of 100 trials each. After each block, there was a mandatory break period lasting 1 minute. 

The data included here correspond to the main experiment only (i.e. practice and calibration data are not included).

In the data analyzed in the paper (n=20), subject 14 was excluded from analysis due to using the highest confidence rating on > 96% of trials, precluding meaningful calculation of meta-d'. The excluded subject is included in this data set of n=21.


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
NaN --> subject failed to enter response within 2 sec of stimulus offset

* Confidence
1 - 4 --> subject entered confidence 1 - 4
NaN --> subject failed to enter confidence within 2 sec of stimulus offset, OR confidence was not collected on this trial

* RT_dec
number --> seconds between stimulus onset and entry of response
NaN --> subject failed to enter response within 2 sec of stimulus offset

* RT_conf
number --> seconds between entry of response and entry of confidence
NaN --> subject failed to enter confidence within 2 sec of stimulus offset, OR confidence was not collected on this trial

* Contrast
Michelson contrast of grating

* BlockNumber
block of trials in which current trial was contained

* BlockType
0 --> partial type 2 block; confidence not collected for first 50 trials of the block
1 --> whole type 2 block; confidence collected for all 100 trials of the block

* ConfCollected
0 --> confidence not collected on this trial
1 --> confidence collected on this trial