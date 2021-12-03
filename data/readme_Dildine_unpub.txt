Contributors: Troy C. Dildine, Lauren Y. Atlas

Contributors emails: troy.dildine@nih.gov, lauren.atlas@nih.gov

Citation: These data are not published yet.

Sample: Healthy volunteers between 18-50 (“Subj_idx” in our spreadsheet), were drawn from a community sample from Washington DC and neighboring regions. Additional demographics are available upon request.

Experiment dates: October 2015 - August 2016

Experiment setting: Participants received stimulation in a behavioral testing room within an out-patient clinic. 

Experiment goal: The main goal of the procedure was to establish pain tolerance and evaluate the reliability of the relationship between temperature and pain. A second goal was to measure whether we can measure confidence in subjective pain rating using explicit and implicit measures. 

Manipulation: Participants received temperatures calibrated to elicit low, medium, and high pain using an individualized adaptive staircase procedure. There were no additional manipulations. 

Block: 24 trials of heat stimulation (“Trial_number” in our spreadsheet) were applied to 8 sites of the forearm ("Skin_site" in our spreadsheet) in three rounds. 

Feedback: No feedback was given to the participant; however, due to safety and ethical concerns we did check in with participants periodically to verify the participant understood the scale and that a temperature was tolerable (please see note about RT corrections).   

Response device: A mouse was used to make ratings throughout the task. Mouse position was restricted to the scale boundaries (horizontal movement only). 

Stimulus: Participants experienced 24 trials of noxious heat delivered via a 16x16mm thermode (Medoc Ltd., Ramat Yisha, Israel) at varying temperatures ranging from 36-50 degrees Celsius (“Stimulus” in our spreadsheet). Heat was applied to the left volar forearm of the participant.
 
Rating: Following heat offset (when the temperature returned to baseline levels), participants provided pain ratings (“Response” in our spreadsheet). Pain ratings were recorded using a mouse on a visual analogue scale from 0 to 10 that only appeared on a computer screen after heat offset with the following anchors:  0 (no sensation), warm but not painful trials (1), pain threshold (2), moderate pain (5), maximum tolerable pain (8), and 10 (most imaginable pain). If a response was coded below 0 or above 10, the mouse position was verified and the rating was coded zero or 10 accordingly. Original values for these trials are also available (“raw_Response” in our spreadsheet). 

Confidence scale: Following the pain rating on each trial, participants were asked to rate how certain they were in their pain ratings (“Confidence” in our spreadsheet). Although the  question probes confidence, we anchored and used the term uncertainty to simplify for our participants. Our 0-100 visual uncertainty scale was anchored as (0) zero uncertainty and (100) complete uncertainty. We REVERSED scored the raw ratings, to parallel other confidence scales (i.e., 100 is associated with complete confidence). If a raw response was coded below zero or above 100 by the computer, the mouse position was verified, and the rating was recoded as zero or 100 accordingly. We determined that a response of zero on the scale could be captured by 11.15 pixels or 0.8% of the scale (8 pixels for width of the scale border and 3.15 for width of the arrow). Therefore, any response below 0.8 would be coded as zero in our model. We also re-coded any trial coded as NaN in our corrected reaction time measures (“logTransformed_RT_dec_NaNOutlier” and “logTransformed_RT_conf_NaNOutlier” in our spreadsheet; see details below) as NaN for confidence.   

Reaction Time Decision: This RT measure refers to the reaction time for pain ratings (“RT_dec” in our spreadsheet), not for the confidence/uncertainty ratings. Participants were presented the pain rating scale after heat offset and had 3 seconds to look at the pain scale prior to being able to use their mouse to record a rating. They had no time limit to make their rating. 

Reaction Time Confidence: This RT measure refers to the reaction time for the confidence rating (“RT_conf” in our spreadsheet), not for the pain rating. Participants were presented the confidence scale directly after rating pain. Participants used a mouse to make their ratings and they had no time limit to make their rating. 

Raw confidence: Our 0-100 visual uncertainty scale was anchored as (0) zero uncertainty and (100) complete uncertainty.  The scale was placed 200 pixels higher on the screen compared to the pain rating to prevent orienting from the pain rating scale. 0 on the scale = no uncertainty, so please note that the original scores (“raw_Confidence_NOTReverseCoded” in our spreadsheet) are the inverse of confidence.  
  
log Transformed RT decision: We log transformed RT data because of non-normality. We also removed trials that were more than 3 std from the mean for each subject. These trials often occurred if subjects asked for clarification on pain ratings, leading to longer than normal response times. Removing these trials and associated confidence ratings is advised (“logTransformed_RT_dec_NaNOutlier” in our spreadsheet). 

log Transformed RT confidence: We log transformed RT data because of non-normality. We also removed trials that were more than 3 std from the mean for each subject. These trials often occurred if subjects or experimenters asked for clarification on pain ratings, to ensure safety and understanding of the scale. This led to longer than normal response times. Removing these trials and the associated confidence ratings is advised (“logTransformed_RT_conf_NaNOutlier” in our spreadsheet).