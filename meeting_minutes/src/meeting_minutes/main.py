#!/usr/bin/env python
from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pathlib import Path
from dotenv import load_dotenv
import os
import agentops

from crews.meeting_minutes_crew.meeting_crew import MeetingMinutesCrew
from crews.gmailcrew.gmailcrew  import GmailCrew


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
        print("Generating Meeting Minutes")

        crew = MeetingMinutesCrew()

        inputs = {
            "transcript": self.state.transcript
        }
        meeting_minutes = crew.crew().kickoff(inputs)
        self.state.meeting_minutes = meeting_minutes

    @listen(generate_meeting_minutes)
    def create_draft_meeting_minutes(self):
        print("Creating Draft Meeting Minutes")

        crew = GmailCrew()

        inputs = {
             "body": str(self.state.meeting_minutes.output)
        }

        draft_crew = crew.crew().kickoff(inputs)
        print(f"Draft Crew: {draft_crew}")


def kickoff():
    session = agentops.init(api_key=os.getenv("AGENTOPS_API_KEY"))
    meeting_minutes_flow = MeetingMinutesFlow()
    meeting_minutes_flow.plot()
    meeting_minutes_flow.kickoff()
    session.end_session()
if __name__ == "__main__":
    kickoff()