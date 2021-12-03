readme_Matthews_2019.txt
Contributor: Julian Matthews (julian.r.matthews@gmail.com) 

Citation: Matthews, J., Wu, J., Corneille, V., Hohwy, J., van Boxtel, J., & Tsuchiya, N. (2019). Sustained conscious access to incidental memories in RSVP. Attention, Perception, & Psychophysics, 81(1), 188-204.

Stimulus: An RSVP paradigm. After observing a rapid sequence of greyscale face images at fixation, participants identified which of 2 stimulus alternatives was a probe face that appeared in the sequence (probe recognition task). Stimulus corresponds to whether the probe face appeared on the left (code: ‘probe_left’) or right (code: ‘probe_right’) of the response screen. Response indicates the side the participant selected. A very large pool of face stimuli were used to ensure participants never (or very rarely) saw a stimulus twice.

Confidence scale: Participants made a simultaneous decision and confidence rating. Confidence was rated on a 4-point scale from 1 to 4. Participants were instructed that a rating of 1 corresponded to a ‘complete guess’ and a rating of 4 corresponded to ‘absolutely certain’. No explicit verbal instructions were given for confidence ratings of 2 or 3 but participants were encouraged to use the full confidence scale.

Manipulations: We employed a mixed design with 5 manipulations: Memory type, Probe lag, Trial length, Scene type, and Stimulus orientation. Memory type corresponded to whether participants were required to use incidental or explicit memory to perform the probe recognition task. In the incidental memory condition (Task code: ‘target_search’) participants were presented with a target face before each RSVP sequence and were required to identify this target face (with a mouse-click) when it appeared in the sequence. If they misidentified or failed to respond within a 700ms window following the target face, they were presented with a startling and aversive error screen, skipping the probe recognition task (Response/Confidence cells are coded as NaNs in these instances). If the target was accurately identified, participants proceeded to the probe recognition task but had to rely on their incidental memory of the non-target distractor faces from the sequence. In the explicit memory condition (Task code: ‘no_target_search’) participants viewed each RSVP sequence without searching for a target. Participants performed the probe recognition task on every trial. Probe lag was manipulated within-subjects and corresponded to the location of the probe face relative to either the target (in ‘target_search’ trials) or the last face in the sequence (in ‘no_target_search’ trials). Trial length was manipulated within-subjects and corresponded to the length of the RSVP sequence (from 8 to 15 items) (Trial_length codes 8 to 15). Scene type corresponded to whether the face stimuli in each trial sequence were cropped from the same group photograph or from different photographs (Condition_Scene codes ‘within_scene’ and ‘across_scene’). Stimulus orientation corresponded to whether the face stimuli were presented upright or inverted (Condition_Orientation codes ‘upright’ and ‘inverted’). 

Block size: Each block was 40 trials in length. Participants completed 8 blocks per condition. A maximum of 4 blocks were completed each day of testing. Participants were tested a minimum of 2 days (with some returning for multiple sessions: see Subject population). 

Feedback: Participants received no trial-by-trial feedback on the probe recognition task.

NaN fields: Stimulus/Response/Confidence/etc cells are coded ‘NaN’ if participants failed to identify the target face in ‘target_search’ blocks. Unfortunately, no reaction time data was recorded (probe recognition judgments were untimed). For completion sake, I have included a variable (RT_decConf) but coded it NaN.

Subject population: Age range 19-31. Participants returned for several sessions. To highlight this I have included the participant code rather than participant number in the Subject_idx column.

Response device: Mouse.

Experimental setting: In lab supervised by experimenter.

Special instructions: Subjects were encouraged to use the full confidence scale.

Link to material/codes: https://github.com/julian-matthews/incidental-memory-RSVP

Experiment dates: 2016.

Location of data collection: Monash Biomedical Imaging, Clayton, AUSTRALIA.

Language of study: English.

Experimenter(s): First and third authors of study, both familiar with psychophysics.