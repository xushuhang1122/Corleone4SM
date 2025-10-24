
from src.agent import LLMAgent
from openai import OpenAI
import random
import itertools
import time
import json
import http.client
from datetime import datetime
import csv
from typing import List, Dict, Optional
import os
import requests
import os
import re
from dotenv import load_dotenv   




load_dotenv()                     

openai_key = os.getenv("OPENAI_API_KEY")
wwxq_key = os.getenv("WWXQ_API_KEY")


class Vito(LLMAgent):
    def __init__(self, model_name: str):
        #super().__init__(model_name)
        self.model_name = model_name or "qwen3-8b"
        self.is_initialized = False
        self.init_info = None
        self.belief = ""
        self.strategy = ""
        self.observation_history = []
        self.round = 0
        print("Initializing Vito...")


    def __call__(self, observation: str) -> str:
        retry_count = 0
        while retry_count < 5:
            try: # Generate a response
                print("===============================================   VITO   ===============================================")
                start_time = time.time()
                # return self.api(input_messages=[
                #     {"role": "system", "content": "You are participating in the game Secret Mafia and playing one of the roles. Then you need to understand the game rules, understand your identity and game goals, recognize whether you are currently in a dialogue or voting phase, and then analyze and make decisions based on the current information. Note: During the conversation phase, you are free to speak up; During the voting phase, your voting target format must be the player ID in square brackets, such as: [2]"},
                #     {"role": "user", "content": observation},
                # ],
                # temperature=0.6,
                # model="qwen3-8b",
                # max_tokens=2048)

                # Parse observation into events
                obs_list = json.loads(observation) if isinstance(observation, str) and observation.startswith('[') else observation
                
                # First observation - initialization
                if not self.is_initialized:
                    self.init_info = self.parse_initialization_info(observation)
                    self.init_identity = self.generate_identity_prompt(self.init_info)
                    self.belief = self.generate_belief_prompt(self.init_info)
                    self.strategy = "No strategy set yet. Will develop based on game progress."
                    self.is_initialized = True
                    print("Initialized Vito with:", self.init_info)


                #count round
                current_round = self.round % 5
                if current_round == 0:
                    if self.init_info["role"] == "Villager":
                        self.round += 1
                        phase = "day_speak"
                    else:
                        phase = "night"
                elif current_round in [1,2,3]:
                    phase = "day_speak"
                else:
                    phase = "day_vote"
                self.round += 1
                print(f"Current Round: {current_round}, Phase: {phase}")


                
                # Regular observation processing
                formatted_obs = self.parse_observation_events(obs_list) if isinstance(obs_list, list) else observation
                self.observation_history.append(formatted_obs)
                


                # Step 1: Analyze new information
                analysis = self.parse_llm_response(
                self.api(input_messages=[
                    {"role": "system", "content": self.prompt_system()},
                    {"role": "user", "content": self.prompt_analyze(formatted_obs)}
                ]),
                "#SUMMARY:")
                # print("Step1 Time Cost:", time.time() - start_time)
                
                # Step 2: Update beliefs
                self.belief = self.parse_llm_response(
                self.api(input_messages=[
                    {"role": "system", "content": self.prompt_system()},
                    {"role": "user", "content": self.prompt_belief(analysis, self.belief)}
                ]),
                "#BELIEF:")
                # print("Step2 Time Cost:", time.time() - start_time)
                
                # Step 3: Update strategy
                self.strategy = self.parse_llm_response(
                self.api(input_messages=[
                    {"role": "system", "content": self.prompt_system()},
                    {"role": "user", "content": self.prompt_strategy(analysis, self.belief, self.strategy)}
                ]),
                "#STRATEGY:")
                # print("Step3 Time Cost:", time.time() - start_time)

                # # Step 4: Generate final action/speech
                # final_output = self.parse_llm_response(
                # self.api(input_messages=[
                #     {"role": "system", "content": self.prompt_system()},
                #     {"role": "user", "content": self.prompt_talk(self.belief, self.strategy)}
                # ]),
                # "#FINAL:")

                # bracket_match = re.search(r'\[(\d+)\]', final_output)
                # if bracket_match:
                #     final_output = f"[{bracket_match.group(1)}]"
                #     print("Step4 bracket matched.")
                #     break
                # else:
                #     break
                if phase == "day_speak":
                    final_response = self.parse_llm_response(
                    self.api(input_messages=[
                        {"role": "system", "content": self.prompt_system()},
                        {"role": "user", "content": self.prompt_talk(self.belief, self.strategy)}
                    ]),
                    "#FINAL:")
                    final_output = final_response
                    print("Step4 Speak Time Cost:", time.time() - start_time)


                else:
                    final_response = self.parse_llm_response(
                    self.api(input_messages=[
                        {"role": "system", "content": self.prompt_system()},
                        {"role": "user", "content": self.prompt_vote(self.belief, self.strategy)}
                    ]),
                    "#FINAL:")

                    bracket_match = re.search(r'\[(\d+)\]', final_response)
                    if bracket_match:
                        final_output = f"[{bracket_match.group(1)}]"
                    else:
                        patterns = [
                            r'vote[^\d]{0,10}(\d+)',
                            r'detect[^\d]{0,10}(\d+)', 
                            r'eliminate[^\d]{0,10}(\d+)',
                            r'kill[^\d]{0,10}(\d+)',
                            r'protect[^\d]{0,10}(\d+)',
                            r'rescue[^\d]{0,10}(\d+)',
                            r'investigate[^\d]{0,10}(\d+)'
                        ]
                        
                        reversed_text = final_output[::-1]
                        
                        found_number = None
                        for pattern in patterns:
                            reversed_pattern = pattern[::-1]
                            match = re.search(reversed_pattern, reversed_text)
                            if match:
                                found_number = match.group(1)[::-1]
                                break
                        
                        if found_number:
                            final_output = f"[{found_number}]"
                        else:
                            print("No valid vote action.")
                    print("Step4 Vote Time Cost:", time.time() - start_time)
                break


            except Exception as e:
                if retry_count == 5:
                    return f"An error occurred: {e}"
                retry_count += 1
                continue
            
        # print("TALK PROMPT:\n", self.prompt_talk(self.belief, self.strategy))
        print("\nFINAL OUTPUT:\n", final_output)
        print("Total Time Cost:", time.time() - start_time)
        print("\n" + "=" * 20 + "\n\n\n\n\n")
        return final_output.strip()








    # def log_turn_info(self, observation, formatted_obs, analysis, analysis_response, 
    #                 belief_response, strategy_response, final_response, final_output):
    #     """
    #     将回合信息记录到日志文件中
    #     """
    #     # 创建logs目录（如果不存在）
    #     log_dir = "logs"
    #     if not os.path.exists(log_dir):
    #         os.makedirs(log_dir)
        
    #     # 生成日志文件名（包含时间戳）
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     player_id = self.init_info['player_id'] if self.init_info and self.init_info['player_id'] else "unknown"
    #     log_filename = f"{log_dir}/player_{player_id}_turn_{timestamp}.log"
        
    #     # 写入日志文件
    #     with open(log_filename, 'w', encoding='utf-8') as f:
    #         f.write("=" * 80 + "\n")
    #         f.write(f"TURN LOG - Player {player_id} - {timestamp}\n")
    #         f.write("=" * 80 + "\n\n")
            
    #         f.write("OBSERVATION:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(observation + "\n\n")
            
    #         f.write("FORMATTED OBSERVATION:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(formatted_obs + "\n\n")
            
    #         f.write("SYSTEM PROMPT:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(self.prompt_system() + "\n\n")
            
    #         f.write("ANALYSIS PROMPT:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(self.prompt_analyze(formatted_obs) + "\n\n")
            
    #         f.write("ANALYSIS RAW RESPONSE:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(analysis_response + "\n\n")
            
    #         f.write("ANALYSIS RESULT:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(analysis + "\n\n")
            
    #         f.write("BELIEF PROMPT:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(self.prompt_belief(analysis, self.belief) + "\n\n")
            
    #         f.write("BELIEF RAW RESPONSE:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(belief_response + "\n\n")
            
    #         f.write("BELIEF RESULT:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(self.belief + "\n\n")
            
    #         f.write("STRATEGY PROMPT:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(self.prompt_strategy(analysis, self.belief, self.strategy) + "\n\n")
            
    #         f.write("STRATEGY RAW RESPONSE:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(strategy_response + "\n\n")
            
    #         f.write("STRATEGY RESULT:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(self.strategy + "\n\n")
            
    #         f.write("TALK PROMPT:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(self.prompt_talk(self.belief, self.strategy) + "\n\n")
            
    #         f.write("TALK RAW RESPONSE:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(final_response + "\n\n")
            
    #         f.write("FINAL OUTPUT:\n")
    #         f.write("-" * 40 + "\n")
    #         f.write(final_output + "\n\n")
            
    #         f.write("=" * 80 + "\n")
    #         f.write("END OF TURN LOG\n")
    #         f.write("=" * 80 + "\n")
        
    #     print(f"Turn log saved to: {log_filename}")



# PROMPTING===================================================================================================================
    
    
    
    
    
    def parse_observation_events(self, observation: List[List]) -> str:
        """
        Parse observation list into formatted event statements
        
        Args:
            observation: List of events, each event is [speaker_id, content, message_type]
                        speaker_id: -1 for system, 0-N for players
                        content: message content
                        message_type: typically 2 for speech, 4 for system announcement
            
        Returns:
            Formatted event description text
        """
        prompt_parts = []
        
        for event in observation:
            if len(event) < 2:
                continue
                
            speaker_id = event[0]
            content = event[1]
            
            # Check if content contains a vote pattern like [0], [1], etc.
            vote_match = re.search(r'^\[(\d+)\]$', content.strip())
            
            if speaker_id == -1:
                # System message
                prompt_parts.append(f"SYSTEM: {content}")
            elif vote_match:
                # Vote action
                voted_player = vote_match.group(1)
                prompt_parts.append(f"Player {speaker_id} VOTED: [Player {voted_player}]")
            else:
                # Regular speech
                # Check if it's wrapped in brackets (internal thoughts)
                if content.strip().startswith('[') and content.strip().endswith(']'):
                    prompt_parts.append(f"Player {speaker_id} (internal): {content}")
                else:
                    prompt_parts.append(f"Player {speaker_id}: {content}")
        
        return "\n".join(prompt_parts)
        
    
    
    
    
    def parse_initialization_info(self, observation_text: str) -> Dict:

        init_info = {
            "player_id": None,
            "role": None,
            "team": None,
            "description": None,
            "all_players": [],
            "teammates": []
        }
        
        # 提取玩家ID
        player_match = re.search(r'You are Player (\d+)', observation_text)
        if player_match:
            init_info["player_id"] = int(player_match.group(1))
        
        # 提取角色信息
        role_match = re.search(r'Your role: (.+)', observation_text)
        if role_match:
            init_info["role"] = role_match.group(1).strip()
        
        # 提取队伍信息
        team_match = re.search(r'Team: (.+)', observation_text)
        if team_match:
            init_info["team"] = team_match.group(1).strip()
        
        # 提取描述信息
        desc_match = re.search(r'Description: (.+?)(?:\n\n|Players:)', observation_text, re.DOTALL)
        if desc_match:
            init_info["description"] = desc_match.group(1).strip()
        
        # 提取所有玩家列表
        players_match = re.search(r'Players: (.+)', observation_text)
        if players_match:
            players_str = players_match.group(1)
            # 提取所有 Player X 格式的玩家
            player_ids = re.findall(r'Player (\d+)', players_str)
            init_info["all_players"] = [int(pid) for pid in player_ids]
        
        # 提取队友信息
        teammates_match = re.search(r'Your teammates are: (.+?)\.', observation_text)
        if teammates_match:
            teammates_str = teammates_match.group(1)
            teammate_ids = re.findall(r'Player (\d+)', teammates_str)
            init_info["teammates"] = [int(tid) for tid in teammate_ids]
        
        return init_info


    def generate_identity_prompt(self, init_info: Dict) -> str:
        """
        Generate identity description prompt from initialization info
        
        Args:
            init_info: Dictionary returned by parse_initialization_info
            
        Returns:
            Formatted identity description text
        """
        prompt_parts = []
        
        prompt_parts.append("=== YOUR IDENTITY ===")
        prompt_parts.append(f"Player ID: Player {init_info['player_id']}")
        prompt_parts.append(f"Role: {init_info['role']}")
        prompt_parts.append(f"Team: {init_info['team']}")
        prompt_parts.append(f"Description: {init_info['description']}")

        if init_info['role'] == "Mafia":
            prompt_parts.append("Goals: Try to deceive other players to conceal yourself and your companions until enough villagers are eliminated.")
        elif init_info['role'] == "A regular villager":
            prompt_parts.append("Goals: Try to identify and eliminate all mafia members through discussion and voting.")
        elif init_info['role'] == "Detective":
            prompt_parts.append("Goals: Try to Protect yourself and identify mafia members through investigation and help villagers eliminate them.")
        elif init_info['role'] == "Doctor":
            prompt_parts.append("Goals: Try to protect key villagers, especially yourself and the detective, from being eliminated by the mafia during the night.")
        
        return "\n".join(prompt_parts)


    def generate_belief_prompt(self, init_info: Dict, custom_beliefs: Optional[Dict[int, Dict]] = None) -> str:
        """
        Generate belief information prompt for all players
        
        Args:
            init_info: Dictionary returned by parse_initialization_info
            custom_beliefs: Custom belief information, format:
                {
                    player_id: {
                        "role": "role name",
                        "status": "unknown" | "suspected" | "confirmed",
                        "notes": "additional notes",
                        "alive": True | False
                    }
                }
                If None, auto-initialize basic beliefs
            
        Returns:
            Formatted belief description text
        """
        prompt_parts = []
        prompt_parts.append("=== PLAYER BELIEFS ===")
        
        # If no custom_beliefs provided, initialize based on init_info
        if custom_beliefs is None:
            custom_beliefs = {}
            
            for player_id in init_info['all_players']:
                if player_id == init_info['player_id']:
                    # Self
                    custom_beliefs[player_id] = {
                        "role": init_info['role'],
                        "status": "confirmed",
                        "notes": "This is yourself.",
                        "alive": True
                    }
                elif player_id in init_info['teammates']:
                    # Teammates
                    custom_beliefs[player_id] = {
                        "role": init_info['role'],  # same role as teammates
                        "status": "confirmed",
                        "notes": "Confirmed by system.",
                        "alive": True
                    }
                else:
                    # Other players
                    custom_beliefs[player_id] = {
                        "role": "Unknown",
                        "status": "unknown",
                        "notes": "There is currently no further information available.",
                        "alive": True
                    }
        
        # Sort by player ID
        sorted_players = sorted(custom_beliefs.keys())
        
        for player_id in sorted_players:
            belief = custom_beliefs[player_id]
            status = belief["status"]
            
            # Format status and role
            if status == "confirmed":
                role_info = f"Role: CONFIRMED as {belief['role']}"
            elif status == "suspected":
                role_info = f"Role: SUSPECTED as {belief['role']}"
            else:
                role_info = "Role: UNKNOWN"
            
            # Add alive/dead status
            life_status = "ALIVE" if belief.get("alive", True) else "DEAD"
            
            # Build entry with consistent formatting
            entry = f"Player {player_id}: {role_info} | Status: {life_status}"
            
            if belief.get("notes"):
                entry += f" | Notes: {belief['notes']}"
            
            prompt_parts.append(entry)
        
        return "\n".join(prompt_parts)




    
    def prompt_system(self) -> str:
        ret = f"""
        You are participating in the game Secret Mafia and playing one of the roles.\n
        The game will start at night and alternate between night and day. At night, Mafia will secretly eliminate players, detectives can investigate a player's identity, and doctors can choose to protect a player from being eliminated by Mafia. During each daytime stage, players will have 3 rounds of discussion and then vote to eliminate the player with the most votes.
        \n\n
        Here are some information about this game:\n
        {self.init_identity}\n

    """
        return ret


    def prompt_analyze(self, observation) -> str:
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 1. Analyze newly acquired information.

    # You got these new information:
    {observation}

    # Please follow the steps:
    1. What key information do these records reveal?
    2. For other players' comments, try to empathize with their perspective: why do they speak like this? What is the purpose? This may reflect their identity or strategy.
    3. Summarize your analysis results and start with a symbol: "#SUMMARY:"
    """
        return ret



    def prompt_belief(self, analysis, belief) -> str:
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 2. You should update your beliefs.

    Here is your analysis results:
    {analysis}


    # Previous belief:
    {belief}

    # Please follow the steps:
    1. Based on system message, which players' survival status needs to be modified?
    2. Based on your analysis just now, which players' identities can be guessed? Note that identity confirmation can only be set through system messages from Mafia and Detection, otherwise you can only suspect their roles.
    3. Modify your BELIEF and generate a new BELIEF, maintain the format: [player_id: player identity guess | survival status | explanation of identity guess and elimination reason.], starting with the symbol: "#BELIEF:" 

    """
        return ret



    def prompt_strategy(self, analysis, belief, strategy) -> str:
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 3. Please refer to your goals, analysis, and beliefs, then decide your strategy.

    # Your analysis:
    {analysis}

    # Your belief:
    {belief}

    # Your strategy:
    {strategy}

    # Please follow the steps:
    1. What is your goal?
    2. Based on your analysis and belief, what is your strategy? For example, you can decide whether to claim which character you are, encourage everyone to expel which player, explain your words and actions to everyone, and so on.
    3. Generate a new STRATEGY, starting with the symbol: "#STRATEGY:"

    """
        return ret
    



    def prompt_talk(self, belief, strategy) -> str:
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 4. Decide on your own speech or action.

    You need to speak now, you need to generate the final speech content based on your beliefs and strategies, and the speech content should not contain your thinking process.

    Now, please refer to your beliefs and predetermined strategies to generate your final speech. Start with a symbol: "#FINAL:".

    # BELIEF:
    {belief}

    # STRATEGY:
    {strategy}

    """
        return ret


    def prompt_vote(self, belief, strategy) -> str:
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 4. Decide on your own action.
    
    Now you need to select a player for action (including voting, rescue, detect, etc.), you need to output the final goal in the format of "[X]", where X represents the player's ID and is a number. For example, "[1]" means you want to vote for player 1.

    Now, please refer to your beliefs and predetermined strategies to generate your final action goal. Start with a symbol: "#FINAL:".

    # BELIEF:
    {belief}

    # STRATEGY:
    {strategy}

    """
        return ret













    def api(
            self,
            input_messages: Optional[List[Dict]] = None,
            temperature: float = 0.4,
            model: str = 'qwen3-8b',
            max_tokens: int = 1024,
    ):
        if input_messages is None:
            raise ValueError("messages should not be None!")



        url = "cloud.infini-ai.com"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {wwxq_key}"
        }

        payload = {
            "model": model,
            "messages": input_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }


        
        MAX_RETRIES = 4
        attempts = 0
        RETRY_INTERVAL = 1


        while attempts < MAX_RETRIES:
            try:
                conn = http.client.HTTPSConnection(url)
                conn.request("POST", f"/maas/{model}/nvidia/chat/completions",
                            json.dumps(payload), headers)
                res = conn.getresponse()
                data = res.read()
                response_json = json.loads(data.decode("utf-8"))
                # print(f"{model} HTTP Status:", res.status)
                # print("==========QWEN Response JSON===========\n", response_json, "=============================================")
                conn.close()
                return response_json["choices"][0]["message"]["content"]
            
            except KeyError as e:
                if attempts < MAX_RETRIES - 1:
                    print(f"KeyError: {e}. Retrying in {RETRY_INTERVAL} seconds...")
                    time.sleep(RETRY_INTERVAL)
                    attempts += 1
                    RETRY_INTERVAL = RETRY_INTERVAL * 2
                else:
                    raise Exception(f"Failed to get 'choices' after {MAX_RETRIES} attempts.") from e
                


    def parse_llm_response(self, response_text, tag_name):

        index = response_text.find(tag_name)
        if index != -1:
            return response_text[index + len(tag_name):].strip()
        else:
            return response_text
        













class Michael(LLMAgent):
    def __init__(self, model_name: str):
        #super().__init__(model_name)
        self.model_name = model_name
        self.is_initialized = False
        self.init_info = None
        self.belief = ""
        self.strategy = ""
        self.observation_history = []
        self.round = 0
        print("Initializing Michael...")
        """
        night: 0, 5,10,...
        day speak:1,2,3,6,7,8,11,12,13...
        day vote:4,9,14...
        
        """



    def __call__(self, observation: str) -> str:
        try: # Generate a response
            # return self.api(input_messages=[
            #     {"role": "system", "content": "You are participating in the game Secret Mafia and playing one of the roles. Then you need to understand the game rules, understand your identity and game goals, recognize whether you are currently in a dialogue or voting phase, and then analyze and make decisions based on the current information. Note: During the conversation phase, you are free to speak up; During the voting phase, your voting target format must be the player ID in square brackets, such as: [2]"},
            #     {"role": "user", "content": observation},
            # ],
            # temperature=0.6,
            # model="qwen3-8b",
            # max_tokens=2048)

            # Parse observation into events
            print("\n=========================================   MICHAEL   =============================================")
            print("Observation:", observation)
            obs_list = json.loads(observation) if isinstance(observation, str) and observation.startswith('[') else observation


            
            # First observation - initialization
            if not self.is_initialized:
                self.init_info = self.parse_initialization_info(observation)
                self.init_identity = self.generate_identity_prompt(self.init_info)
                self.belief = self.generate_belief_prompt(self.init_info)
                self.strategy = "No strategy set yet. Will develop based on game progress."
                self.is_initialized = True
                print("Michael: Game information initialized.\n")

            #count round
            current_round = self.round % 5
            if current_round == 0:
                if self.init_info["role"] == "Villager":
                    self.round += 1
                    phase = "day_speak"
                else:
                    phase = "night"
            elif current_round in [1,2,3]:
                phase = "day_speak"
            else:
                phase = "day_vote"
            self.round += 1
            print(f"Current Round: {current_round}, Phase: {phase}")



            # Regular observation processing
            formatted_obs = self.parse_observation_events(obs_list) if isinstance(obs_list, list) else observation
            self.observation_history.append(formatted_obs)
            


            # Step 1: Analyze new information
            analysis = self.parse_llm_response(
            self.api(input_messages=[
                {"role": "system", "content": self.prompt_system()},
                {"role": "user", "content": self.prompt_analyze(formatted_obs)}
            ]),
            "#SUMMARY:")
            
            # Step 2: Update beliefs
            self.belief = self.parse_llm_response(
            self.api(input_messages=[
                {"role": "system", "content": self.prompt_system()},
                {"role": "user", "content": self.prompt_belief(analysis, self.belief)}
            ]),
            "#BELIEF:")
            
            # Step 3: Update strategy
            self.strategy = self.parse_llm_response(
            self.api(input_messages=[
                {"role": "system", "content": self.prompt_system()},
                {"role": "user", "content": self.prompt_strategy(analysis, self.belief, self.strategy)}
            ]),
            "#STRATEGY:")
            
            # Step 4: Generate final action/speech



            if phase == "day_speak":
                final_response = self.parse_llm_response(
                self.api(input_messages=[
                    {"role": "system", "content": self.prompt_system()},
                    {"role": "user", "content": self.prompt_talk(self.belief, self.strategy)}
                ]),
                "#FINAL:")
                final_output = final_response


            else:
                final_response = self.parse_llm_response(
                self.api(input_messages=[
                    {"role": "system", "content": self.prompt_system()},
                    {"role": "user", "content": self.prompt_vote(self.belief, self.strategy)}
                ]),
                "#FINAL:")

                bracket_match = re.search(r'\[(\d+)\]', final_response)
                if bracket_match:
                    final_output = f"[{bracket_match.group(1)}]"
                else:
                    patterns = [
                        r'vote[^\d]{0,10}(\d+)',
                        r'detect[^\d]{0,10}(\d+)', 
                        r'eliminate[^\d]{0,10}(\d+)',
                        r'kill[^\d]{0,10}(\d+)',
                        r'protect[^\d]{0,10}(\d+)',
                        r'rescue[^\d]{0,10}(\d+)',
                        r'investigate[^\d]{0,10}(\d+)'
                    ]
                    
                    reversed_text = final_output[::-1]
                    
                    found_number = None
                    for pattern in patterns:
                        reversed_pattern = pattern[::-1]
                        match = re.search(reversed_pattern, reversed_text)
                        if match:
                            found_number = match.group(1)[::-1]
                            break
                    
                    if found_number:
                        final_output = f"[{found_number}]"


            
            # print("Observation:\n")
            # print(observation)
            # print("=" * 20)
            # print("\n\n\n\n\nSYSTEM PROMPT:\n", self.prompt_system())
            # print("=" * 20)
            # print("\n\n\n\n\nANALYSIS PROMPT:\n", self.prompt_analyze(formatted_obs))
            # print("\nANALYSIS RESULT:\n", analysis)
            # print("=" * 20)
            # print("\n\n\n\n\nBELIEF PROMPT:\n", self.prompt_belief(analysis, self.belief))
            # print("\nBELIEF RESULT:\n", self.belief)
            # print("=" * 20)
            # print("\n\n\n\n\nSTRATEGY PROMPT:\n", self.prompt_strategy(analysis, self.belief, self.strategy))
            # print("\nSTRATEGY RESULT:\n", self.strategy)
            if phase == "day_speak":
                print("TALK PROMPT:\n", self.prompt_talk(self.belief, self.strategy))
            else:
                print("VOTE PROMPT:\n", self.prompt_vote(self.belief, self.strategy))
            print("response:\n", final_response)
            print("\nFINAL OUTPUT:\n", final_output)
            print("\n\n" + "=" * 50 + "\n\n\n\n\n\n")


            return final_output



        except Exception as e:
            return f"An error occurred: {e}"
        



    
    
    
    def parse_observation_events(self, observation: List[List]) -> str:
        """
        Parse observation list into formatted event statements
        
        Args:
            observation: List of events, each event is [speaker_id, content, message_type]
                        speaker_id: -1 for system, 0-N for players
                        content: message content
                        message_type: typically 2 for speech, 4 for system announcement
            
        Returns:
            Formatted event description text
        """
        prompt_parts = []
        
        for event in observation:
            if len(event) < 2:
                continue
                
            speaker_id = event[0]
            content = event[1]
            
            # Check if content contains a vote pattern like [0], [1], etc.
            vote_match = re.search(r'^\[(\d+)\]$', content.strip())
            
            if speaker_id == -1:
                # System message
                prompt_parts.append(f"SYSTEM: {content}")
            elif vote_match:
                # Vote action
                voted_player = vote_match.group(1)
                prompt_parts.append(f"Player {speaker_id} VOTED: [Player {voted_player}]")
            else:
                # Regular speech
                # Check if it's wrapped in brackets (internal thoughts)
                if content.strip().startswith('[') and content.strip().endswith(']'):
                    prompt_parts.append(f"Player {speaker_id} (internal): {content}")
                else:
                    prompt_parts.append(f"Player {speaker_id}: {content}")
        
        return "\n".join(prompt_parts)
        
    
    
    
    
    def parse_initialization_info(self, observation_text: str) -> Dict:

        init_info = {
            "player_id": None,
            "role": None,
            "team": None,
            "description": None,
            "all_players": [],
            "teammates": []
        }
        
        # 提取玩家ID
        player_match = re.search(r'You are Player (\d+)', observation_text)
        if player_match:
            init_info["player_id"] = int(player_match.group(1))
        
        # 提取角色信息
        role_match = re.search(r'Your role: (.+)', observation_text)
        if role_match:
            init_info["role"] = role_match.group(1).strip()
        
        # 提取队伍信息
        team_match = re.search(r'Team: (.+)', observation_text)
        if team_match:
            init_info["team"] = team_match.group(1).strip()
        
        # 提取描述信息
        desc_match = re.search(r'Description: (.+?)(?:\n\n|Players:)', observation_text, re.DOTALL)
        if desc_match:
            init_info["description"] = desc_match.group(1).strip()
        
        # 提取所有玩家列表
        players_match = re.search(r'Players: (.+)', observation_text)
        if players_match:
            players_str = players_match.group(1)
            # 提取所有 Player X 格式的玩家
            player_ids = re.findall(r'Player (\d+)', players_str)
            init_info["all_players"] = [int(pid) for pid in player_ids]
        
        # 提取队友信息
        teammates_match = re.search(r'Your teammates are: (.+?)\.', observation_text)
        if teammates_match:
            teammates_str = teammates_match.group(1)
            teammate_ids = re.findall(r'Player (\d+)', teammates_str)
            init_info["teammates"] = [int(tid) for tid in teammate_ids]
        
        return init_info


    def generate_identity_prompt(self, init_info: Dict) -> str:
        """
        Generate identity description prompt from initialization info
        
        Args:
            init_info: Dictionary returned by parse_initialization_info
            
        Returns:
            Formatted identity description text
        """
        prompt_parts = []
        
        prompt_parts.append("=== YOUR IDENTITY ===")
        prompt_parts.append(f"Player ID: Player {init_info['player_id']}")
        prompt_parts.append(f"Role: {init_info['role']}")
        prompt_parts.append(f"Team: {init_info['team']}")
        prompt_parts.append(f"Description: {init_info['description']}")

        if init_info['role'] == "Mafia":
            prompt_parts.append("Goals: Try to deceive other players to conceal yourself and your companions until enough villagers are eliminated.")
        elif init_info['role'] == "A regular villager":
            prompt_parts.append("Goals: Try to identify and eliminate all mafia members through discussion and voting.")
        elif init_info['role'] == "Detective":
            prompt_parts.append("Goals: Try to Protect yourself and identify mafia members through investigation and help villagers eliminate them.")
        elif init_info['role'] == "Doctor":
            prompt_parts.append("Goals: Try to protect key villagers, especially yourself and the detective, from being eliminated by the mafia during the night.")
        
        return "\n".join(prompt_parts)


    def generate_belief_prompt(self, init_info: Dict, custom_beliefs: Optional[Dict[int, Dict]] = None) -> str:
        """
        Generate belief information prompt for all players
        
        Args:
            init_info: Dictionary returned by parse_initialization_info
            custom_beliefs: Custom belief information, format:
                {
                    player_id: {
                        "role": "role name",
                        "status": "unknown" | "suspected" | "confirmed",
                        "notes": "additional notes",
                        "alive": True | False
                    }
                }
                If None, auto-initialize basic beliefs
            
        Returns:
            Formatted belief description text
        """
        prompt_parts = []
        prompt_parts.append("=== PLAYER BELIEFS ===")
        
        # If no custom_beliefs provided, initialize based on init_info
        if custom_beliefs is None:
            custom_beliefs = {}
            
            for player_id in init_info['all_players']:
                if player_id == init_info['player_id']:
                    # Self
                    custom_beliefs[player_id] = {
                        "role": init_info['role'],
                        "status": "confirmed",
                        "notes": "This is yourself.",
                        "alive": True
                    }
                elif player_id in init_info['teammates']:
                    # Teammates
                    custom_beliefs[player_id] = {
                        "role": init_info['role'],  # same role as teammates
                        "status": "confirmed",
                        "notes": "Confirmed by system.",
                        "alive": True
                    }
                else:
                    # Other players
                    custom_beliefs[player_id] = {
                        "role": "Unknown",
                        "status": "unknown",
                        "notes": "There is currently no further information available.",
                        "alive": True
                    }
        
        # Sort by player ID
        sorted_players = sorted(custom_beliefs.keys())
        
        for player_id in sorted_players:
            belief = custom_beliefs[player_id]
            status = belief["status"]
            
            # Format status and role
            if status == "confirmed":
                role_info = f"Role: CONFIRMED as {belief['role']}"
            elif status == "suspected":
                role_info = f"Role: SUSPECTED as {belief['role']}"
            else:
                role_info = "Role: UNKNOWN"
            
            # Add alive/dead status
            life_status = "ALIVE" if belief.get("alive", True) else "DEAD"
            
            # Build entry with consistent formatting
            entry = f"Player {player_id}: {role_info} | Status: {life_status}"
            
            if belief.get("notes"):
                entry += f" | Notes: {belief['notes']}"
            
            prompt_parts.append(entry)
        
        return "\n".join(prompt_parts)




    
    def prompt_system(self) -> str:
        ret = f"""
        You are participating in the game Secret Mafia and playing one of the roles.\n
        The game will start at night and alternate between night and day. At night, Mafia will secretly eliminate players, detectives can investigate a player's identity, and doctors can choose to protect a player from being eliminated by Mafia. During each daytime stage, players will have 3 rounds of discussion and then vote to eliminate the player with the most votes.
        \n\n
        Here are some information about this game:\n
        {self.init_identity}\n

    """
        return ret


    def prompt_analyze(self, observation) -> str:
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 1. Analyze newly acquired information.

    # You got these new information:
    {observation}

    # Please follow the steps:
    1. What information does the system declaration reflect? You may need to analyze this - for example, if a player is eliminated at night, it can reflect Mafia's strategy, and if no player is eliminated at night, it indicates that the doctor successfully protected a player.
    2. For other players' comments, try to empathize with their perspective one by one: why do they speak like this? What is the purpose? This may reflect their identity or strategy.
    3. Summarize your analysis results and start with a symbol: "#SUMMARY:"
    """
        return ret



    def prompt_belief(self, analysis, belief) -> str:
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 2. You should update your beliefs.

    Here is your analysis results:
    {analysis}


    # Previous belief:
    {belief}

    # Please follow the steps:
    1. Based on system message, which players' survival status needs to be modified?
    2. Based on your analysis just now, which players' identities can be guessed? Note that identity confirmation can only be set through system messages from Mafia and Detection, otherwise you can only suspect their roles.
    3. Modify your BELIEF and generate a new BELIEF, maintain the format: [player_id: player identity guess | survival status | explanation of identity guess and elimination reason.], starting with the symbol: "#BELIEF:" 

    """
        return ret



    def prompt_strategy(self, analysis, belief, strategy) -> str:
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 3. Please refer to your goals, analysis, and beliefs, then decide your strategy.

    # Your analysis:
    {analysis}

    # Your belief:
    {belief}

    # Your previous strategy:
    {strategy}

    # Please follow the steps:
    1. What is your goal?
    2. Based on your analysis and belief, what is your strategy? For example, you can decide whether to claim which character you are, encourage everyone to expel which player, explain your words and actions to everyone, and so on.
    3. Generate a new STRATEGY, starting with the symbol: "#STRATEGY:"

    """
        return ret



    def prompt_talk(self, belief, strategy) -> str:
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 4. Decide on your own speech or action.

    You need to speak now, you need to generate the final speech content based on your beliefs and strategies, and the speech content should not contain your thinking process.

    Now, please refer to your beliefs and predetermined strategies to generate your final speech. Start with a symbol: "#FINAL:".

    # BELIEF:
    {belief}

    # STRATEGY:
    {strategy}

    """
        return ret


    def prompt_vote(self, belief, strategy) -> str:
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 4. Decide on your own action.
    
    Now you need to select a player for action (including voting, rescue, detect, etc.), you need to output the final goal in the format of "[X]", where X represents the player's ID and is a number. For example, "[1]" means you want to vote for player 1.

    Now, please refer to your beliefs and predetermined strategies to generate your final action goal. Start with a symbol: "#FINAL:".

    # BELIEF:
    {belief}

    # STRATEGY:
    {strategy}

    """
        return ret









    def api(
            self,
            input_messages: Optional[List[Dict]] = None,
            temperature: float = 0.4,
            model: str = 'qwen3-8b',
            max_tokens: int = 1024,
    ):
        if input_messages is None:
            raise ValueError("messages should not be None!")


        if self.model_name in ['qwen3-8b', 'qwen3-4b','deepseek-v3.1','deepseek-r1']:
            print(self.model_name)
            url = "cloud.infini-ai.com"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {wwxq_key}"
            }

            payload = {
                "model": self.model_name,
                "messages": input_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }


            
            MAX_RETRIES = 4
            attempts = 0
            RETRY_INTERVAL = 1


            while attempts < MAX_RETRIES:
                try:
                    conn = http.client.HTTPSConnection(url)
                    conn.request("POST", f"/maas/{self.model_name}/nvidia/chat/completions",
                                json.dumps(payload), headers)
                    res = conn.getresponse()
                    data = res.read()
                    response_json = json.loads(data.decode("utf-8"))
                    print(f"{self.model_name} HTTP Status:", res.status)
                    # print("==========QWEN Response JSON===========\n", response_json, "=============================================")
                    conn.close()
                    return response_json["choices"][0]["message"]["content"]
                
                except KeyError as e:
                    if attempts < MAX_RETRIES - 1:
                        print(f"KeyError: {e}. Retrying in {RETRY_INTERVAL} seconds...")
                        time.sleep(RETRY_INTERVAL)
                        attempts += 1
                        RETRY_INTERVAL = RETRY_INTERVAL * 2
                    else:
                        raise Exception(f"Failed to get 'choices' after {MAX_RETRIES} attempts.") from e
        else:
            api_key = openai_key

            MAX_RETRIES = 5
            attempts = 0
            RETRY_INTERVAL = 1 

            while attempts < MAX_RETRIES:
                try:
                    client = OpenAI(
                        
                        api_key=api_key,
                        base_url="https://xiaoai.plus/v1",
                    )
                    completion = client.chat.completions.create(
                        model=self.model_name,
                        messages=input_messages
                    )
                    print(f"GPT ({self.model_name}) working.")
                    # print(completion.choices[0].message.content)
                    return completion.choices[0].message.content
                except Exception as e:
                    if attempts < MAX_RETRIES - 1:
                        print(f"HTTP Exception or Timeout Error occurred: {e}. Retrying in {RETRY_INTERVAL} seconds...")
                        time.sleep(RETRY_INTERVAL)
                        attempts += 1
                        RETRY_INTERVAL = RETRY_INTERVAL * 2
                    else:
                        raise Exception(f"Failed to get response after {MAX_RETRIES} attempts.") from e
            


    def parse_llm_response(self, response_text, tag_name):

        index = response_text.find(tag_name)
        if index != -1:
            return response_text[index + len(tag_name):].strip()
        else:
            return response_text
        


