Contributor
-----------
Brian Maniscalco
bmaniscalco@gmail.com


Citation
--------
Maniscalco, B., McCurdy, L. Y., Odegaard, B., & Lau, H. (2017). Limited Cognitive Resources Explain a Trade-Off between Perceptual and Metacognitive Vigilance. The Journal of Neuroscience, 37(5), 1213–1224. https://doi.org/10.1523/JNEUROSCI.2271-13.2016

Experiment 4


Experiment details
------------------

(for full details, see the paper cited above)

Circular patches of white noise were presented to the left and right of fixation for 33 ms. A sinusoidal grating was embedded in either the left or right patch of noise. The subject's task was to indicate which patch contained the grating, left or right, with a keypress.

On some (but not all) trials, after entering the stimulus discrimination response, subjects wagered on the accuracy of their response. Subjects wagered points on a scale of 1 to 4, with 4 being highest, with the goal of maximizing their cumulative points at the end of the experiment. Subjects were encouraged to wager more points when confidence was higher and fewer points when confidence was lower.

There was a 2 second period after stimulus offset in which the subject had to enter response and wager. The full 2 seconds elapsed before the start of the next trial even if response and wager were entered early. This brief and mandatory response period was introduced in order to standardize block duration for different block types (see below) to facilitate comparison of decision fatigue effects in the different block types.

In odd numbered blocks ("partial type 2 blocks"), subjects did not provide point wagers for the first 50 of 100 trials in the block; on even numbered blocks ("whole type 2 blocks"), however, they provided point wagers on all trials. (This manipulation was introduced to test a hypothesis about how having to evaluate confidence might influence decision fatigue.) On trials where a point wager was collected, text appeared on the screen after entry of the stimulus discrimination response to prompt entry of the wager. Text also appeared on the screen before each block to indicate to the subject whether wagers would be collected on every trial, or only the last 50 trials.

On trials where wagers were collected, subjects won the number of points wagered for correct responses, and lost the number of points wagered for incorrect responses. On trials where wagers were not collected, subjects won 3 points for correct responses and lost 3 points for incorrect responses. For all trials, failure to enter the response and wager within the 2 second time limit led to a loss of 10 points.

Subjects first performed 2 practice blocks of 28 trials each. Subjects received performance feedback after each trial during practice (high pitched tone following correct responses, low pitched tone for incorrect or missing responses). No trial-by-trial performance feedback was provided for the remainder of the experiment.

Subjects then performed a calibration block of 120 trials in order to determine the grating contrast to be used in the main experiment.

In the main experiment, subjects performed 10 blocks of 100 trials each. After each block, there was a mandatory break period lasting 1 minute. During the break period, subjects were shown how many points they had accrued thus far, as well as how many points would have resulted if they had bet 4 points for every correct response and 1 point for every incorrect response.

The data included here correspond to the main experiment only (i.e. practice and calibration data are not included).

In the data analyzed in the paper (n=27), subjects 3, 4, 14, 25, 26, and 32 were excluded from analysis due to using the highest wager on > 93% of trials, precluding meaningful calculation of meta-d'. The excluded subjects are included in this data set of n=33.


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
1 - 4 --> subject entered wager 1 - 4
NaN --> subject failed to enter wager within 2 sec of stimulus offset, OR wager was not collected on this trial

* RT_dec
number --> seconds between stimulus onset and entry of response
NaN --> subject failed to enter response within 2 sec of stimulus offset

* RT_conf
number --> seconds between entry of response and entry of wager
NaN --> subject failed to enter wager within 2 sec of stimulus offset, OR wager was not collected on this trial

* Contrast
Michelson contrast of grating

* BlockNumber
block of trials in which current trial was contained

* BlockType
0 --> partial type 2 block; wager not collected for first 50 trials of the block
1 --> whole type 2 block; wager collected for all 100 trials of the block

* ConfCollected
0 --> wager not collected on this trial
1 --> wager collected on this trial

* Points
number of points won or lost on this trial