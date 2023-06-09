# Welcome to the Nuggt documentation!

This guide will provide you with the necessary information to start using Nuggt and harness its capabilities for automating tasks using AI agents.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Prompt Template](#prompt-template)
  - [Tool Integration](#tool-integration)
  - [Step-by-Step Execution](#step-by-step-execution)
- [Examples](#examples)
- [Contributing](#contributing)
- [Support](#support)

## Installation

1. Clone the Nuggt repository from GitHub: `git clone https://github.com/Nuggt-dev/Nuggt.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Navigate to the project directory: `cd nuggt-release`
4. Launch streamlit: `streamlit run nuggt.py`

You can setup the API keys either under `.env` file or set it up under settings after launching the app.

## Usage

Nuggt provides a flexible and customizable way to automate tasks. The following sections explain the key aspects of using Nuggt.

### Prompt Template

Nuggt utilizes a step-by-step prompt template to define the actions required for task execution. Each step consists of the following elements:

- Step: A brief description of the current step.
- Reason: The reason behind taking this step.
- Action: The action to be performed, selected from the available tools.
- Action Input: The input required for the action.
- Observation: The result or output obtained from the action.

## Tool Integration

Nuggt offers seamless integration with various tools, allowing you to extend the functionality of your AI agents. The available tools include:

- Python: Execute Python scripts using a Python shell. Syntax: `{tool:python}`
- Document Tool: Process documents and retrieve answers to questions. Syntax: `{tool:document_tool}`
- Video Tool: Analyze videos and extract answers to questions. Syntax: `{tool:video_tool}`
- Google: Perform Google searches and retrieve relevant information. Syntax: `{tool:google}`
- Browse Website: Interact with websites, automate tasks, and extract data. Syntax: `{tool:browse_website}`
- Search: Get answers from google via SerperAPI. Syntax: `{tool:search}`
- LLM: Create LLM instance. Syntax: `{tool:llm}`
- Stable Diffusion: Generate images using Stable Diffusion. Syntax: `{tool:stable_diffusion}`
- Video Generation: Generate video from text. Syntax: `{tool:generate_video}`
- Caption: Caption images using SceneXplainAPI. Syntax: `{tool:image_caption}`
- Display: Print tables/visualisations to console. Syntax: `{tool:display}`


### Step-by-Step Execution

To execute a task using Nuggt, follow these steps:

1. Define a prompt using the following inputs:
   - `{text:<variable>}`: Create an input field and assign its value to `<variable>`
   - `{upload:<variable>}`: Create a file upload button and save the uploaded file by the name `<variable>`
   - `{tool:<tool_name>}`: Specify a tool to use. 

   For example: `Open {upload:document} using {tool:python} and display its content using {tool:display}`

2. Provide the necessary inputs and tools for each step, following the prompt format. Replace `<variable>` with any variable name.

3. Execute the task using Nuggt by clicking the submit button, which will process the prompt and perform the specified actions based on the inputs provided.

4. Review the observations and final answer generated by Nuggt to obtain the desired output.

By utilizing the `{tool:<tool_name>}` input format, you can seamlessly integrate various tools into your prompts, such as Python, document processing, video analysis, Google search, and web browsing. Combine these inputs with `{text:<variable>}` and `{upload:<variable>}` to create dynamic and interactive prompts tailored to your specific requirements.

## Examples

These examples cover a variety of tasks and demonstrate how to structure your prompts for efficient automation.

### Example 1: Website Search and Information Gathering

Prompt: `Find websites related to {text:query} using {tool:google}. Browse three results to gather information on {text:query} using {tool:browse_website}. Display the results in the format <Content: Content of the website, URL: URL of the website> using {tool:display}.`

Description:
This example demonstrates how to use Nuggt to search for websites related to a specific query using Google. Then browse three search results, gather information from those websites using the browse_website tool, and display the results in a specific format using the display tool.

### Example 2: File Analysis and Task Completion

Prompt: `Open {upload:file} using {tool:python}. Display its description using {tool:python}. Complete the task {text:input} using {tool:python}. Display your results using {tool:data_analysis}.`

Description:
In this example, you'll learn how to analyze a file using Nuggt. First, you open the specified file using the python tool, then display its description using the same tool. Next, you complete a task like "Create basic visualisations" using the python tool, and finally display the results using the data_analysis tool.

### Example 3: Stock Data Analysis

Prompt: `Get past 7 days OHLC data for stock {text:stock} using {tool:python}. Calculate the MACD using {tool:python} (Do not use talib). Display the results in a table using {tool:display}. Display basic visualizations using {tool:display}.`

Description:
This example focuses on stock data analysis with Nuggt. First, you retrieve the past 7 days' OHLC (Open, High, Low, Close) data for a specific stock using the python tool. Then, calculate the MACD (Moving Average Convergence Divergence) without using talib library, using the python tool. Display the results in a table format using the display tool, and also showcase basic visualizations using the same display tool.

Feel free to adapt and explore these examples based on your specific use cases and requirements.

## Contributing

We welcome contributions from the community! If you would like to contribute to Nuggt, please refer to the [contributing guidelines](contribution_guidelines.md) for more information on how to get involved.

## Support

If you encounter any issues or have questions regarding Nuggt, please feel free to reach out to our support team at {discord}. We are here to assist you and ensure a smooth experience with Nuggt.

Thank you for choosing Nuggt. Happy automating!
