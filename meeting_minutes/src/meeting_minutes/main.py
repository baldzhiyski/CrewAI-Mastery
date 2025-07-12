#!/usr/bin/env python
from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pathlib import Path
from dotenv import load_dotenv
import os

from crews.meeting_minutes_crew.meeting_crew import MeetingMinutesCrew


load_dotenv()


class MeetingMinutesState(BaseModel):
    transcript: str = ""
    meeting_minutes: str = ""


class MeetingMinutesFlow(Flow[MeetingMinutesState]):

    @start()
    def transcribe_meeting(self):
        print("🎙️ Transcribing audio file...")

        SCRIPT_DIR = Path(__file__).parent
        audio_path = str(SCRIPT_DIR / "EarningsCall.wav")

        # Load the local Whisper model (base/tiny/small/medium/large)
        model = WhisperModel("base", compute_type="int8")

        segments, _ = model.transcribe(audio_path)

        # Join the transcript segments
        full_transcription = " ".join([segment.text for segment in segments])
        self.state.transcript = full_transcription

        print("\n📝 Transcript:")
        print(self.state.transcript + "\n")


    @listen(transcribe_meeting)
    def generate_meeting_minutes(self):
        print("🧠 Generating meeting minutes using CrewAI...")

        crew = MeetingMinutesCrew()

        inputs = {
            "transcript": self.state.transcript
        }
        crew_output = crew.crew().kickoff(inputs)

        # Extract plain result string from CrewOutput
        result_string = crew_output.result

        self.state.meeting_minutes = result_string

        print("\n📄 Meeting Minutes:")
        print(result_string[:1000] + "...\n")  # Print first 1000 characters

    @listen(generate_meeting_minutes)
    def save_meeting_minutes(self):
        print("💾 Saving meeting minutes to file...")

        output_path = Path(__file__).parent / "meeting_minutes.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.state.meeting_minutes)

        print(f"✅ Saved to {output_path.absolute()}")


def kickoff():
    flow = MeetingMinutesFlow()
    flow.plot()
    flow.kickoff()


if __name__ == "__main__":
    kickoff()
