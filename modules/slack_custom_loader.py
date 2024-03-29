import json
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import re 

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

class CustomSlackDirectoryLoader(BaseLoader):
    """Load from a `Slack` directory dump."""

    def __init__(self, zip_path: str, workspace_url: Optional[str] = None):
        """Initialize the SlackDirectoryLoader.

        Args:
            zip_path (str): The path to the Slack directory dump zip file.
            workspace_url (Optional[str]): The Slack workspace URL.
              Including the URL will turn
              sources into links. Defaults to None.
        """
        self.zip_path = Path(zip_path)
        self.workspace_url = workspace_url
        self.channel_id_map = self._get_channel_id_map(self.zip_path)

    @staticmethod
    def _get_channel_id_map(zip_path: Path) -> Dict[str, str]:
        """Get a dictionary mapping channel names to their respective IDs."""
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            try:
                with zip_file.open("channels.json", "r") as f:
                    channels = json.load(f)
                return {channel["name"]: channel["id"] for channel in channels}
            except KeyError:
                return {}

    def load(self, padding=4, name_field="real_name", channels=None) -> List[Document]:
        """Load and return documents from the Slack directory dump.
        
        Parameters

        ----------
        padding : int, optional
        The amount of padding to apply to the document text, by default 4.
        [Explanation of how the padding value affects the loading process.]

        """
        docs = []

        self.users = self.read_users(self.zip_path, name_field=name_field)

        with zipfile.ZipFile(self.zip_path, "r") as zip_file:       
            for channel_path in zip_file.namelist():
                channel_name = Path(channel_path).parent.name
                
                if not channel_name:
                    continue
                
                # Skip if channels if not specified
                if len(channels) > 0 and channel_name not in channels:
                    # print( f'skipping: {channel_path}')
                    continue

                # These are Slack files per channel for each individual day
                if channel_path.endswith(".json"):

                    messages = self._read_json(zip_file, channel_path)
                    messages = self._filter_user_messages('USLACKBOT', messages)

                    # Sort and group messages by thread or time proximity
                    threads = self._group_messages(messages)

                    for thread in threads:
                        date_time = self._dt2str(thread[0].get("ts", 0))
                        user_ids = list(set([message.get("user", "") for message in thread]))
                        username = list(self.users.get(user_id, user_id) for user_id in user_ids)

                        txt = self._thread_to_string(thread, channel_name=channel_name)
                        
                        docs.append(Document(
                            page_content=txt,
                            metadata={
                                "channel_name": channel_name,
                                "type": "Slack Message",
                                "date": date_time,
                                # "users": username, #disabled - chromaDB doesnt like lists
                                "nr_characters": len(txt),
                                "nr_reactions": len(thread)
                            },
                        ))      

        return docs
    
    def _replace_mentions(self, message:str, userlist=[] ) -> str:
        """Replaces mentions with actual names"""

        users = self.users
        # Function to lookup the username by ID
        def replace_mention(match):
            id_ = match.group(1)  # Extract the ID from the match
            return users.get(id_, "unknown")  # Return the username or "unknown" if not found
        
        # Remove mentions
        message = re.sub("<@([A-Z0-9]+)>", replace_mention, message)

        # Remove slack channel references
        message = re.sub("<#([A-Z0-9]+)>", replace_mention, message)

        # remove newlines
        message = message.replace('\n', ' ')

        #remove double spaces
        message = message.replace('  ', ' ')

        return message
    
    @staticmethod
    def _clean_string( message:str ) -> str:
        """Cleaning up some data"""

        #Clear newlines
        message = message.replace('\n', ' ')

        # Remove emojies like :wine_glass:
        message = re.sub(r':[a-zA-Z0-9_+-]+:', '', message)

        # Removing <!here> mentinoe
        message = re.sub("<!.*?>", "", message)

        # TODO: REemove USLACKBOT mentions (optional)

        # TODO: We should do something with hypterlinks.

        return message
    
    @staticmethod
    def _group_messages(messages: List[dict], max_interval: int = 30) -> List[List[dict]]:
        """Group messages if it's a thread or messages posted within 15 minutes of each other."""
        
        sorted_messages = sorted(messages, key=lambda x: float(x["ts"]))
        all_threads = []
        current_thread = []

        for message in sorted_messages:
          
            if( len(current_thread) == 0 ):
                current_thread.append(message)
                continue

            if( len(current_thread) > 0 ):

                if( message.get("thread_ts", False) ):
                    # Create a new thread if the message is part of a thread
                    if( message.get("thread_ts", False) == current_thread[-1].get("thread_ts", True) ):
                       current_thread.append(message)
                    else:
                        all_threads.append(current_thread)
                        current_thread = [message]

                else: 
                    # If the message is within x minutes we consider it a reply and add it to a thread.
                    last_message = current_thread[-1]
                    last_timestamp = datetime.utcfromtimestamp(float(last_message["ts"]))
                    timestamp = datetime.utcfromtimestamp(float(message["ts"]))

                    if( timestamp - last_timestamp < timedelta(minutes=max_interval) ):
                        current_thread.append(message)
                    else:
                        all_threads.append(current_thread)
                        current_thread = [message]

        # Close the last thread
        if current_thread:
            all_threads.append(current_thread)

        return all_threads

    @staticmethod
    def _dt2str(ts: datetime) -> str:
        """Convert a timestamp to a string."""
        dt = datetime.fromtimestamp( float( ts ) )

        # Making this an int - for easier filtering (later)
        return int(dt.strftime('%Y%m%d%H%M%S')) 
    
    @staticmethod
    def _filter_user_messages(user_id: str, messages: List[dict]) -> List[dict]:
        """Filter out messages from a specific user."""
        return [message for message in messages if message.get("user", "") != user_id]
    
    def _thread_to_string(
        self, thread: dict, channel_name: str
    ) -> Document:
        """Convert a Slack thread to a single text string."""

        # This should be put in the metadata.
        # text = f"--- On {self._dt2str( thread[0].get('ts',0) )} in {channel_name} ---\n"
        text = ""

        for i, message in enumerate(thread):   
            user_id = message.get("user", "")
            username = self.users.get(user_id, user_id)
            
            # Keeping it with fist-names to make the vecor db cleaner
            # We might need to skip it altogether for the search iteslf
            # And then retrieve the "original" document to parse to the LLM.
            username = username.split(' ')[0]
            if( i == 0):
                text += f"{username}: " 
            if( i > 0):
                text += f"{username}: "
            text += message.get("text", "") + "\n"
        
        # Replace mentions
        text = self._replace_mentions(text)
        text = self._clean_string(text)
        return text
            
    def _read_json(self, zip_file: zipfile.ZipFile, file_path: str) -> List[dict]:
        """Read JSON data from a zip subfile."""
        
        with zip_file.open(file_path, "r") as f:
            data = json.load(f)
        return data

    @staticmethod
    def read_users(zip_path, name_field="display_name") -> List[dict]:
        """Read User JSON file and returns a list with user names and ids"""
        with zipfile.ZipFile( zip_path, "r") as zip_file:
            with zip_file.open('users.json') as user_json:
                data = json.load(user_json)
                names_per_id = {entry['id']: entry.get('profile').get(name_field) for entry in data}
                return names_per_id

    def _convert_message_to_document(
        self, message: dict, channel_name: str
    ) -> Document:
        """
        Convert a message to a Document object.

        Args:
            message (dict): A message in the form of a dictionary.
            channel_name (str): The name of the channel the message belongs to.

        Returns:
            Document: A Document object representing the message.
        """
        text = message.get("text", "")
        metadata = self._get_message_metadata(message, channel_name)
        return Document(
            page_content=text,
            metadata=metadata,
        )

    def _get_message_metadata(self, message: dict, channel_name: str) -> dict:
        """Create and return metadata for a given message and channel."""
        timestamp = message.get("ts", "")
        user_id = message.get("user", "")

        user = self.users.get(user_id, user_id) 

        source = self._get_message_source(channel_name, user, timestamp)
        return {
            "source": source,
            "channel": channel_name,
            "timestamp": timestamp,
            "user": user,
        }

    def _get_message_source(self, channel_name: str, user: str, timestamp: str) -> str:
        """
        Get the message source as a string.

        Args:
            channel_name (str): The name of the channel the message belongs to.
            user (str): The user ID who sent the message.
            timestamp (str): The timestamp of the message.

        Returns:
            str: The message source.
        """
        if self.workspace_url:
            channel_id = self.channel_id_map.get(channel_name, "")
            return (
                f"{self.workspace_url}/archives/{channel_id}"
                + f"/p{timestamp.replace('.', '')}"
            )
        else:
            return f"{channel_name} - {user} - {timestamp}"