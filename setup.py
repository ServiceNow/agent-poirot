from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="agentpoirot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="Issam H. Laradji, Gaurav Sahu, Amirhossein Abbasi, Mohammad Chegini",
    author_email="issam.laradji@servicenow.com, gaurav.sahu@uwaterloo.ca",
    description="AgentPoirot: End-to-End Data Analytics Agent",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ServiceNow/research-skilled-poirot",
)
