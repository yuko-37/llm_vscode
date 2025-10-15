import modal
from myagents.agent import Agent

class SpecialistAgent(Agent):
    def __init__(self):
        Pricer = modal.Cls.from_name("pricer-service", "Pricer")
        self.pricer = Pricer()

    def price(self, description: str) -> float:
        reply = self.pricer.price.remote(description)
        return reply