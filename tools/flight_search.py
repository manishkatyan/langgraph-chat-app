import os
import time
import requests
from datetime import datetime, timedelta
from typing import Optional
from langchain_core.runnables.config import RunnableConfig
from langchain.tools import BaseTool
from core import tracer

class FlightSearchTool(BaseTool):
    name: str = "flight_search"
    description: str = """Search for flights between two cities using IATA city codes (e.g., AMS for Amsterdam, ATL for Atlanta). 
    Input should be a string with format 'from ORIGIN to DESTINATION' or 'from ORIGIN to DESTINATION on YYYY-MM-DD'.
    If date is not specified, will search for flights 1 week from today."""
    
    client_id: str = os.getenv("AMADEUS_CLIENT_ID")
    client_secret: str = os.getenv("AMADEUS_CLIENT_SECRET")
    token: Optional[str] = None
    token_expiry: int = 0
    
    # Common IATA city codes for reference
    city_codes: dict[str, str] = {
        'AMS': 'Amsterdam',
        'ATL': 'Atlanta',
        'CLE': 'Cleveland',
        'DEL': 'Delhi',
        # Add more as needed
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def _get_token(self):
        if self.token and time.time() < self.token_expiry:
            return self.token
            
        url = os.getenv("AMADEUS_API_URL") + "/v1/security/oauth2/token"
        payload = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'grant_type': 'client_credentials'
        }
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        
        response = requests.post(url, data=payload, headers=headers)
        data = response.json()
        
        self.token = data['access_token']
        self.token_expiry = time.time() + data['expires_in']
        return self.token

    def _get_default_date(self):
        """Return date 1 week from today in YYYY-MM-DD format"""
        future_date = datetime.now() + timedelta(days=7)
        return future_date.strftime('%Y-%m-%d')

    @tracer.tool(name="flight_search")
    def _run(self, input_str: str, config: Optional[RunnableConfig] = None) -> str:
        try:
            print(f"\n***[LOG] Flight search input: {input_str}\n***")
            # Parse input string
            parts = input_str.lower().split()
            origin_idx = parts.index('from') + 1
            dest_idx = parts.index('to') + 1
            
            origin = parts[origin_idx].upper()
            destination = parts[dest_idx].upper()
            
            # Check if date is specified, otherwise use default
            try:
                date_idx = parts.index('on') + 1
                date = parts[date_idx]
            except ValueError:
                date = self._get_default_date()
            
            # Validate IATA codes
            if len(origin) != 3 or len(destination) != 3:
                return "Please use 3-letter IATA city codes (e.g., AMS for Amsterdam, ATL for Atlanta)"
            
            # Get token and make API call
            token = self._get_token()
            print(f"[LOG] Flight search token: {token}")
            url = os.getenv("AMADEUS_API_URL") + "/v2/shopping/flight-offers"
            headers = {'Authorization': f'Bearer {token}'}
            params = {
                'originLocationCode': origin,
                'destinationLocationCode': destination,
                'departureDate': date,
                'adults': 1,
                'max': 5
            }
            
            print(f"[LOG] Flight search params: {params}")  # Debug logging
            response = requests.get(url, headers=headers, params=params)
            flights = response.json()
            print(f"[LOG] Flight search response: {flights}")
            
            # Format response
            if 'data' in flights:
                result = f"Found flights from {origin} to {destination} for {date}:\n\n"
                for flight in flights['data']:
                    price = flight['price']['total']
                    segments = flight['itineraries'][0]['segments']
                    departure = segments[0]['departure']['at']
                    arrival = segments[-1]['arrival']['at']
                    carriers = [segment['carrierCode'] for segment in segments]
                    
                    result += f"- Flight for ${price}\n"
                    result += f"  Carrier(s): {', '.join(carriers)}\n"
                    result += f"  Departure: {departure}\n"
                    result += f"  Arrival: {arrival}\n"
                    if len(segments) > 1:
                        result += f"  Stops: {len(segments) - 1}\n"
                    result += "\n"
                return result
            else:
                return f"No flights found from {origin} to {destination} on {date}."
                
        except Exception as e:
            return f"Error searching for flights: {str(e)}"