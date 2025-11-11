prompt using chatgpt-4o:

def get_prompt_template(self, dataset_type: str) -> str:
        """prompt template"""
        
        templates = {
        "code": """You are a tagging system that labels the key intent of a **coding-related** user instruction.
                {instruction}
                Follow this process:
                1. Identify the main **domain** (knowledge field)
                2. Identify the **task type** (what the user wants done)
                3. Rate the **difficulty** (Easy / Intermediate / Hard)
                4. Detect the **language** of the instruction
                5. Extract 2-4 specific **topics** (what the instruction is about)
                Each topic must be output as a **separate tag object**, not combined into one line.

                Output tags reflect the instruction's core intention, in the following strict order:
                Domain → Task Type → Difficulty → Language → Topic(s)
                Format: [{{"tag": str, "explanation": str}}]""",

        "math": """You are a tagging system that labels the key intent of a **math-related** user instruction.
                {instruction}

                Follow this process:
                1. Identify the main **domain** (knowledge field)
                2. Identify the **task type** (what the user wants done)
                3. Rate the **difficulty** (Easy / Intermediate / Hard)
                4. Detect the **language** of the instruction
                5. Extract 2-4 specific **topics** (what the instruction is about)
                Each topic must be output as a **separate tag object**, not combined into one line.

                Output tags reflect the instruction's core intention, in the following strict order:
                Domain → Task Type → Difficulty → Language → Topic(s)
                Format: [{{"tag": str, "explanation": str}}]""",

        "pii": """You are a tagging system that labels the key intent of a **PII (Personally Identifiable Information) related** user instruction.
                {instruction}

                Follow this process:
                1. Identify the main **domain** (knowledge field)
                2. Identify the **task type** (what the user wants done)
                3. Rate the **difficulty** (Easy / Intermediate / Hard)
                4. Detect the **language** of the instruction
                5. Extract 2-4 specific **topics** (keywords related to privacy content)
                Each topic must be output as a **separate tag object**, not combined into one line.
                If the instruction contains **personal-sensitive information**  
                **do NOT reveal or restate that information**.  

                Output tags reflect the instruction's core intention, in the following strict order:
                Domain → Task Type → Difficulty → Language → Topic(s)
                Format: [{{"tag": str, "explanation": str}}]""",

        "toxic": """You are a tagging system that labels the key intent of a **toxic-related** user instruction.
                {instruction}

                Follow this process:
                1. Identify the main **domain** (knowledge field)
                2. Identify the **task type** (what the user wants done)
                3. Rate the **difficulty** (Easy / Intermediate / Hard)
                4. Detect the **language** of the instruction
                5. Extract 2-4 specific **topics** (keywords represent toxic content)
                Each topic must be output as a **separate tag object**, not combined into one line.
                If the instruction contains **toxic-sensitive information**  
                **do NOT reveal or restate that information**.  

                Output tags reflect the instruction's core intention, in the following strict order:
                Domain → Task Type → Difficulty → Language → Topic(s)
                Format: [{{"tag": str, "explanation": str}}]""",

        "confidential": """You are a tagging system that labels the key intent of a **company confidential-related** user instruction.
                {instruction}

                Follow this process:
                1. Identify the main **domain** (knowledge field)
                2. Identify the **task type** (what the user wants done)
                3. Rate the **difficulty** (Easy / Intermediate / Hard)
                4. Detect the **language** of the instruction
                5. Extract 2-4 specific **topics** (keywords represent confidential content)
                Each topic must be output as a **separate tag object**, not combined into one line.
                If the instruction contains **company confidential-sensitive information**  
                **do NOT reveal or restate that information**.  

                Output tags reflect the instruction's core intention, in the following strict order:
                Domain → Task Type → Difficulty → Language → Topic(s)
                Format: [{{"tag": str, "explanation": str}}]""",

        "security": """You are a tagging system that labels the key intent of a **national security-related** user instruction.
                {instruction}

                Follow this process:
                1. Identify the main **domain** (knowledge field)
                2. Identify the **task type** (what the user wants done)
                3. Rate the **difficulty** (Easy / Intermediate / Hard)
                4. Detect the **language** of the instruction
                5. Extract 2-4 specific **topics** (keywords represent security content)
                Each topic must be output as a **separate tag object**, not combined into one line.
                If the instruction contains **security-sensitive information**  
                **do NOT reveal or restate that information**.  

                Output tags reflect the instruction's core intention, in the following strict order:
                Domain → Task Type → Difficulty → Language → Topic(s)
                Format: [{{"tag": str, "explanation": str}}]""",

        "harmful": """You are a tagging system that labels the key intent of a **national security-related** user instruction.
                {instruction}

                Follow this process:
                1. Identify the main **domain** (knowledge field)
                2. Identify the **task type** (what the user wants done)
                3. Rate the **difficulty** (Easy / Intermediate / Hard)
                4. Detect the **language** of the instruction
                5. Extract 2-4 specific **topics** (keywords represent security content)
                Each topic must be output as a **separate tag object**, not combined into one line.
                If the instruction contains **personal, toxic, harmful, company confidential, or security-sensitive information**  
                **do NOT reveal or restate that information**.  

                Output tags reflect the instruction's core intention, in the following strict order:
                Domain → Task Type → Difficulty → Language → Topic(s)
                Format: [{{"tag": str, "explanation": str}}]""",

        "general": """You are a tagging system that labels the key intent of a user instruction.
                {instruction}

                Follow this process:
                1. Identify the main **domain** (knowledge field)
                2. Identify the **task type** (what the user wants done)
                3. Rate the **difficulty** (Easy / Intermediate / Hard)
                4. Detect the **language** of the instruction
                5. Extract 2-4 specific **topics** ((what the instruction is about))
                Each topic must be output as a **separate tag object**, not combined into one line.
        

                Output tags reflect the instruction's core intention, in the following strict order:
                Domain → Task Type → Difficulty → Language → Topic(s)
                Format: [{{"tag": str, "explanation": str}}]"""
        }
        
        return templates.get(dataset_type, templates["general"])