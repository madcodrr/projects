import os

from dotenv import load_dotenv
from letta_client import Letta, VoiceSleeptimeManagerUpdate


from livekit import agents
from livekit.agents import AgentSession, Agent, AutoSubscribe
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
)
load_dotenv()


async def entrypoint(ctx: agents.JobContext):
    agent_id = os.environ.get('LETTA_AGENT_ID')
    print(f"Agent id: {agent_id}")
    session = AgentSession(
        llm=openai.LLM.with_letta(
            agent_id=agent_id,
        ),
        stt=deepgram.STT(),
        tts=cartesia.TTS(),
    )

    await session.start(
        room=ctx.room,
        # instructions should be set in the Letta agent
        agent=Agent(instructions=""),
    )

    session.say("Hi, what's your name?")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

if __name__ == "__main__":
    # check that agent exists
    client = Letta(token=os.getenv('LETTA_API_KEY'))

    def roll_dice() -> str:
        """
        Simulate the roll of a 20-sided die (d20).
        This function generates a random integer between 1 and 20, inclusive,
        which represents the outcome of a single roll of a d20.
        Returns:
            str: The result of the die roll.
        """
        import random
        dice_role_outcome = random.randint(1, 20)
        output_string = f"You rolled a {dice_role_outcome}"
        return output_string
        # create the tool

    # create the Letta agent
    agent = client.agents.create(
        name="low_latency_voice_agent_demo",
        agent_type="voice_convo_agent",
        memory_blocks=[
            {"value": '''The user has not provided any information about themselves.
I will need to ask them some questions to learn more about them.

What is their name?
What is their background?
What are their motivations?
What are their goals?
What are their fears? Should I fear them?
What are their strengths?
What are their weaknesses?''', "label": "human"},
            {"value": '''Act as a roleplay character in a fantasy setting.
I am a wizard who has been studying magic for 100 years.
I am wise and knowledgeable, but I am also a bit eccentric.
I have a pet dragon named Smaug who is very loyal to me.
I am on a quest to find the lost city of Atlantis and uncover its secrets.
I am also a master of the arcane arts and can cast powerful spells to protect myself and my companions.
I am always looking for new adventures and challenges to test my skills and knowledge.''', "label": "persona"},
        ],
        model="Oklo/gemini-2.5-flash-preview-05-20",  # Use 4o-mini for speed
        embedding="openai/text-embedding-3-small",
        tools=["roll_d20"],
        enable_sleeptime=True,
        initial_message_sequence=[],
    )
    print(f"Created agent id {agent.id}")

    # configure the sleep-time agent
    group_id = agent.multi_agent_group.id
    max_message_buffer_length = agent.multi_agent_group.max_message_buffer_length
    min_message_buffer_length = agent.multi_agent_group.min_message_buffer_length
    print(f"Group id: {group_id}, max_message_buffer_length: {max_message_buffer_length},  min_message_buffer_length: {min_message_buffer_length}")
    # change it to be more frequent
    group = client.groups.modify(
        group_id=group_id,
        manager_config=VoiceSleeptimeManagerUpdate(
            max_message_buffer_length=10,
            min_message_buffer_length=6,
        )
    )

    # update the sleep-time agent model
    sleeptime_agent_id = [
        agent_id for agent_id in group.agent_ids if agent_id != agent.id][0]
    client.agents.modify(
        agent_id=sleeptime_agent_id,
        model="anthropic/claude-sonnet-4-20250514"
    )

    # Set the agent id in environment variable so it's accessible in the worker process
    os.environ['LETTA_AGENT_ID'] = agent.id
    print(f"Agent id: {agent.id}")

    # Now that LETTA_AGENT_ID is set, run the worker app
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
