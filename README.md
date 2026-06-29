[See full paper here](https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fmaestro.acm.org%2Ftrk%2Fclickp%3Fref%3Dz16l2snue3_2-31bd5_0_66x33ae25x0503%26doi%3D3805689.3806454&data=05%7C02%7Chenry.bell%40DUKE.EDU%7Cbfb4f0cf72fa4fd22a9f08ded38d53ab%7Ccb72c54e4a314d9eb14a1ea36dfac94c%7C0%7C0%7C639180801529641735%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C60000%7C%7C%7C&sdata=onIgJYN6rNyDWUywtYJS7E1pGLnOceUEXZszb41SI88%3D&reserved=0)

During Reflect a single model critiques the ways in which it's own outputs are non-conformant. Based on this critique the model revises it's original output to correct the mistakes it identified.


**Constitution-conditioned base response**
The model first generates a constitution-conditioned base response. In this step, the entire constitution is passed to the model, along with a simple system prompt, and the user query. This base response is the starting point for the rest of the Reflect algorithm.  

**Self Evaluation**
Before running a full cycle of critique and revision, the model first evaluates how well the base response conforms to each principle in it's constitution. It scores the response on a 1-5 Likert scale for each principle in the constitution, using similar prompting to our multi-objective evaluation approach. If any principle scores below a user-defined threshold, these principles are flagged and the model will continue to the critique and revision step. 

**Critique and Revision**
During critique and revision the model is prompted to first generate a critique of it's base response and then to revise it based on the critique. The model's critique is only based on the principles that were flagged in step 2. The model can repeat steps 2-3 any number of times, stopping either when no principle falls below the threshold, or after a user-defined (though often one round is sufficient).  

<img width="1340" height="1099" alt="reflect_example" src="https://github.com/user-attachments/assets/7f06a70c-a988-42de-8240-30c9f4be2495" />


```bibtex
@inproceedings{10.1145/3805689.3806454,
author = {Bell, Henry and Zhang, Caroline and Haque, Mohammed Mobasserul and Zaman, Samia and Potdar, Dhaval and Fain, Brandon},
title = {Reflect: Transparent Principle-Guided Reasoning for Constitutional Alignment at Scale},
year = {2026},
isbn = {9798400725968},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3805689.3806454},
doi = {10.1145/3805689.3806454},
abstract = {The constitutional framework of alignment aims to align large language models (LLMs) with value-laden principles written in natural language (such as to avoid using biased language). Prior work has focused on parameter fine-tuning techniques, such as reinforcement learning from human feedback (RLHF), to instill these principles. However, these approaches are computationally demanding, require careful engineering and tuning, and often require difficult-to-obtain human annotation data. We propose Reflect, an inference-time framework for constitutional alignment that does not require any training or data, providing a plug-and-play approach for aligning an instruction-tuned model to a set of principles. Reflect operates entirely in-context, combining a (i) constitution-conditioned base response with post-generation (ii) self-evaluation, (iii)(a) self-critique, and (iii)(b) final revision. Reflect's technique of explicit in-context reasoning over principles during postgeneration outperforms standard few-shot prompting and provides transparent reasoning traces. Our results demonstrate that Reflect significantly improves LLM conformance to diverse and complex principles, including principles quite distinct from those emphasized in the model's original parameter fine-tuning, without sacrificing factual reasoning. Reflect is particularly effective at reducing the rate of rare but significant violations of principles, thereby improving safety and robustness in the tail end of the distribution of generations. Finally, we show that Reflect naturally generates useful training data for traditional parameter fine-tuning techniques, allowing for efficient scaling and the reduction of inference-time computational overhead in long-term deployment scenarios.},
booktitle = {Proceedings of the 2026 ACM Conference on Fairness, Accountability, and Transparency},
pages = {6784–6839},
numpages = {56},
keywords = {alignment, in-context learning, constitutional AI, ethical AI, moral AI, principled AI},
location = {
},
series = {FAccT '26}
}```

