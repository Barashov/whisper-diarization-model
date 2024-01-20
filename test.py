import time

def test(file_url,
         file_path,
         prompt,
         offset_seconds,
         group_segments):
    diarization_list = None
    segments = None


    # Initialize variables to keep track of the current position in both lists
    margin = 0.1  # 0.1 seconds margin

    # Initialize an empty list to hold the final segments with speaker info
    final_segments = []

    speaker_idx = 0
    n_speakers = len(diarization_list)

    # Iterate over each segment
    for segment in segments:
        segment_start = segment['start'] + offset_seconds
        segment_end = segment['end'] + offset_seconds
        segment_text = []
        segment_words = []

        # Iterate over each word in the segment
        for word in segment['words']:
            word['word'] = word['word'].strip()
            word_start = word['start'] + offset_seconds - margin
            word_end = word['end'] + offset_seconds + margin

            while speaker_idx < n_speakers:
                turn, _, speaker = diarization_list[speaker_idx]

                if turn.start <= word_end and turn.end >= word_start:
                    segment_text.append(word['word'].strip(
                    ))  # Strip the spaces before appending
                    segment_words.append(word)

                    if turn.end <= word_end:
                        speaker_idx += 1

                    break
                elif turn.end < word_start:
                    speaker_idx += 1
                else:
                    break

        if segment_text:
            new_segment = {
                'start': segment_start - offset_seconds,
                'end': segment_end - offset_seconds,
                'speaker': speaker,
                'text': ' '.join(segment_text).strip(),
                'words': segment_words
            }
            final_segments.append(new_segment)
    time_merging_end = time.time()
    print(
        f"Finished with merging, took {time_merging_end - time_diraization_end:.5} seconds"
    )
    segments = final_segments
    # Make output
    output = []  # Initialize an empty list for the output

    # Initialize the first group with the first segment
    current_group = {
        'start': str(segments[0]["start"]),
        'end': str(segments[0]["end"]),
        'speaker': segments[0]["speaker"],
        'text': segments[0]["text"],
        'words': segments[0]["words"]
    }

    for i in range(1, len(segments)):
        # Calculate time gap between consecutive segments
        time_gap = segments[i]["start"] - segments[i - 1]["end"]

        # If the current segment's speaker is the same as the previous segment's speaker, and the time gap is less than or equal to 2 seconds, group them
        if segments[i]["speaker"] == segments[
            i - 1]["speaker"] and time_gap <= 2 and group_segments:
            current_group["end"] = str(segments[i]["end"])
            current_group["text"] += " " + segments[i]["text"]
            current_group["words"] += segments[i]["words"]
        else:
            # Add the current_group to the output list
            output.append(current_group)

            # Start a new group with the current segment
            current_group = {
                'start': str(segments[i]["start"]),
                'end': str(segments[i]["end"]),
                'speaker': segments[i]["speaker"],
                'text': segments[i]["text"],
                'words': segments[i]["words"]
            }

    # Add the last group to the output list
    output.append(current_group)

    time_cleaning_end = time.time()
    print(
        f"Finished with cleaning, took {time_cleaning_end - time_merging_end:.5} seconds"
    )
    time_end = time.time()
    time_diff = time_end - time_start

    system_info = f"""Processing time: {time_diff:.5} seconds"""
    print(system_info)
    return output


import replicate
import os
os.environ['REPLICATE_API_TOKEN'] = 'r8_LV32EpNAW7gG8b8nbt3QckyCBel4rsU2akpKS'
deployment = replicate.deployments.get("alexmarkelov-nh/diarization")
prediction = deployment.predictions.create(
  input={"file_url": "https://replicate.delivery/pbxt/JcL0ttZLlbchC0tL9ZtB20phzeXCSuMm0EJNdLYElgILoZci/AI%20should%20be%20open-sourced.mp3"}
)
prediction.wait()
print(prediction.output)