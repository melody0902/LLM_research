import argparse
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator, FundamentalSuccessRateCalculator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import numpy as np
import random
from utils.timer import timer
from visualize.font_settings import FontSettings
from visualize.visualizer import DiscreteVisualizer
from visualize.legend_settings import DiscreteLegendSettings
from visualize.page_layout_settings import PageLayoutSettings
from visualize.color_scheme import ColorSchemeForDiscreteVisualization
from watermark.kgw.kgw import KGW
from watermark.signature.ngram import KGWNGramSignature


# Setting random seed for reproducibility
seed = 30
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_transformes_config():
    # Transformers config
    # model_name = 'facebook/opt-1.3b'
    model_name = 'meta-llama/Llama-3.1-8B'
    # model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    # model_name = 'taide/Llama3-TAIDE-LX-8B-Chat-Alpha1'
    print(f"使用模型: {model_name}")

    if model_name == 'facebook/opt-1.3b':
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

        transformers_config = TransformersConfig(
                model=model,
                tokenizer=tokenizer,
                vocab_size=50272,
                device=device,
                max_new_tokens=200,
                min_length=230,
                no_repeat_ngram_size=4,
                do_sample=True,
                eos_token_id=None,
            )
    else:    
        # Config for loading the model with quantization
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=nf4_config,
                low_cpu_mem_usage=True,
                local_files_only=True
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

        transformers_config = TransformersConfig(
                model=model,
                tokenizer=tokenizer,
                vocab_size=len(list(tokenizer.get_vocab().values())),
                device=device,
                max_new_tokens=200,
                min_length=230,
                no_repeat_ngram_size=4,
                do_sample=True,
                eos_token_id=None,
            )
    return model, tokenizer, transformers_config

def signature_visualize(myWatermark, watermarked_text, unwatermarked_text):
    # Get data for visualization
    watermarked_data = myWatermark.get_data_for_visualization(watermarked_text)
    unwatermarked_data = myWatermark.get_data_for_visualization(unwatermarked_text)

    # Init visualizer
    visualizer = DiscreteVisualizer(color_scheme=ColorSchemeForDiscreteVisualization(prefix_color='#000000', red_token_color='#CC0000', green_token_color='#006400'),
                                    font_settings=FontSettings(font_path="font/msjh.ttf", font_size=20), 
                                    page_layout_settings=PageLayoutSettings(),
                                    legend_settings=DiscreteLegendSettings())
    # Visualize
    watermarked_img = visualizer.visualize(data=watermarked_data, 
                                        show_text=True, 
                                        visualize_weight=True, 
                                        display_legend=True)    
    
    unwatermarked_img = visualizer.visualize(data=unwatermarked_data,
                                            show_text=True, 
                                            visualize_weight=True, 
                                            display_legend=True)
    # Save
    watermarked_img.save("KGW_signature_watermarked_d0.5.png")
    unwatermarked_img.save("KGW_signature_unwatermarked_d0.5.png")

def standard_visualize(myWatermark, watermarked_text, unwatermarked_text, delta):
    # Get data for visualization
    watermarked_data = myWatermark.get_data_for_visualization(watermarked_text)
    unwatermarked_data = myWatermark.get_data_for_visualization(unwatermarked_text)

    # Init visualizer
    visualizer = DiscreteVisualizer(color_scheme=ColorSchemeForDiscreteVisualization(prefix_color='#000000', red_token_color='#CC0000', green_token_color='#006400'),
                                    font_settings=FontSettings(font_path="font/msjh.ttf", font_size=20), 
                                    page_layout_settings=PageLayoutSettings(),
                                    legend_settings=DiscreteLegendSettings())
    # Visualize
    watermarked_img = visualizer.visualize(data=watermarked_data, 
                                        show_text=True, 
                                        visualize_weight=True, 
                                        display_legend=True)

    unwatermarked_img = visualizer.visualize(data=unwatermarked_data,
                                            show_text=True, 
                                            visualize_weight=True, 
                                            display_legend=True)
    # Save
    watermarked_img.save(f"KGW_watermarked_d{delta}.png")
    unwatermarked_img.save(f"KGW_unwatermarked_d{delta}.png")


if __name__ == "__main__":
    model, tokenizer, transformers_config = get_transformes_config()

    delta = 0.5
    myWatermark = AutoWatermark.load('KGW', 
                                    algorithm_config='config/KGW.json',
                                    transformers_config=transformers_config,
                                    delta=delta
                                    )
    myWatermark.config.prefix_length = 0

    ## zh
    # myWatermark = KGWNGramSignature(
    #                 algorithm_config=f'config/KGW.json',
    #                 transformers_config=transformers_config,
    #                 signature_file='tables_data_100/llama3.1/kgw/zhc4_d2/2-gram/signature_set.json',
    #                 n=2,
    #                 ngram_signature_file='tables_data_100/llama3.1/kgw/zhc4_d2/2-gram/ngram2_signature_set.json'
    #             )

    # watermarked_text = "味料，就能夠為我們帶來一道全新的美味料理。他是如何做的呢？\n　　首先，他選擇了一隻雞腿，然後用刀剖開了它的背部。然後，他再把雞腿放在烤箱中烤熟。他把雞肉上的脂肪擦掉，然後把雞片分成四個小塊。然後他把雞塊放入平底鍋中，將它們煎熟後再把它們移到另一道烤盤上烤熟。在這個過程中，他時時關注著雞塖的色澤。雞塌的外表迅速地轉成金色。\n　　這道美餐是由一隻雛雞腿和香蔥制成的。雞腿烤熟後再煎"

    # unwatermarked_text = "味料，就為他成名發跡的三星米其林餐廳蒙地卡羅巴黎酒店路易十五—艾倫‧杜卡斯餐廳創造了一道優質美饌。如何在尊重食材原味，不過度修飾的原則下，創作出一道與眾不同的米其林料理？以下名廚示範大廚絕不能錯過！\n芫荽洗淨去梗，研缽內放入檸檬汁並研磨芫荽後再加進橄欖油混合備用。\n(1) 胡蘿蔔洗淨削皮，以蔬果榨汁機搾汁後，以篩孔極細的濾網過濾出胡蘿蔔汁。\n(2) 續將胡蘿蔔汁"

    ## en
    # myWatermark = KGWNGramSignature(
    #                 algorithm_config=f'config/KGW.json',
    #                 transformers_config=transformers_config,
    #                 n=2,
    #                 ngram_signature_file='tables_data_1000/llama3.1/kgw/enc4_d0.5/2-gram/ngram2_signature_set.json',
    #             ) 
    
    # watermarked_text_d2 = " it is not. Governments, non-profits, and public institutions all engage in marketing to attract customers, donors, and constituents. In fact, some of the most effective marketers are public sector marketers, who have to navigate complex bureaucracies, limited budgets, and multiple stakeholders to achieve their goals.\nHere are some examples of public sector marketing:\n1. Government tourism marketing: A local government might create a marketing campaign to attract tourists to their area, highlighting its natural beauty, historical sites, and recreational activities.\n2. Public health education: A government health department might develop a marketing campaign aimed at educating the public about the dangers of smoking or the importance of vaccination.\n3. Municipal economic development: A city might create a branding campaign to attract new businesses and residents to its downtown area, emphasizing its vibrant culture, diverse neighborhoods, and access to transportation.\n4. Non-profit fundraising: A charity might develop a direct mail campaign to solicit donations, or create a social media campaign to raise awareness about its cause"
    # watermarked_text_d1 = " it is not. Governments, non-profit organizations, and other public sector entities also employ marketers to develop and execute marketing strategies.\nPublic sector marketers may work for government agencies, non-profit institutions, or other organizations that serve the public interest.\nPublic sector marketing is often focused on issues like public health, education, and economic development, rather than profit maximization.\nPublic sector marketer roles may include developing marketing campaigns to raise awareness about government programs or services, promoting tourism, or encouraging citizens to adopt healthy behaviors.\nPublic sector organizations may also use marketing techniques to build their reputation, engage with stakeholders, and manage their brand.\nPublic sector entities may use marketing strategies such as social media, events, and advertising to reach their target audiences.\nPublic sector agencies may use marketing to build partnerships with other organizations, foster community engagement, and promote economic development.\nPublic sector institutions may use marketing research to understand the needs and preferences of their stakeholders and inform their marketing decisions.\nPublic sector employers may include government agencies, such as the US"
    # watermarked_text_d0_8 = " it is not. Governments, non-profit organizations, and other public sector entities also employ marketers to develop and execute marketing strategies. In fact, public sector marketing is a rapidly growing field, and many governments and non-profits are increasingly recognizing the importance of effective marketing to achieve their goals.\nPublic sector marketers work in a wide range of roles, including:\nGovernment agencies: Marketers work in government agencies to develop and implement marketing campaigns that promote public health, safety, and welfare. For example, a marketer might develop a campaign to encourage people to get vaccinated against flu or to promote road safety.\nNon-profit organizations: Marketers in non-profit organizations develop and execute campaigns that raise awareness and funds for social causes. For example a marketer might work for a charity to develop a campaign that raises awareness about poverty and encourages people to donate.\nPublic utilities: Marketers working in public utilities, such as water and electricity companies, develop and execute customer outreach and education programs to promote the use of these services.\nEducational"
    watermarked_text_d0_5 = " it is also present in the public service. Public sector marketers work for government agencies, non-profit organizations, and other public institutions. Their goal is to promote the services, products, and policies of their organization to the public.\nPublic sector marketers often work in a variety of roles, such as communications specialists, public affairs officers, and policy analysts. They use many of the same marketing techniques as private sector marketers, including research, advertising, public relations, and social media.\nHowever, public sector marketers often face unique challenges, such as limited budgets, strict regulations, and a need to balance competing priorities. They must also be sensitive to the needs and concerns of their constituents, and ensure that their marketing efforts are transparent and accountable.\nSome examples of public sector marketers include:\nGovernment agencies, such as the US Department of Health and Human Services, which uses marketing to promote healthy behaviors and disease prevention.\nNon-profit organizations, such as charities and advocacy groups, which use marketing to raise awareness and funds for their causes"

    unwatermarked_text = " the reality is people with sales and marketing backgrounds are hired by government agencies in a number of capacities. Government agencies at the local, state and federal level all employ marketing professionals in areas including, but not limited to, public relations, property disposal, bond sales and purchasing.\nAlmost all major government agencies have their own public-relations staff, and in many cases it is a stand-alone department with a public relations or media director and several support staff. Government agency PR departments are responsible for producing news releases, holding press conferences, and generally promoting activities of the agency, such as tourism or encouraging new businesses to move into the area.\nGovernment agencies are constantly buying supplies, equipment and other property and selling off old equipment and property. The departments tasked with disposing of this government property often hire individuals with a background in marketing. Their job is to assist the agency in coming up with creative ways to sell or otherwise dispose of obsolete government property.\nMost government agencies have to follow complicated regulations for purchasing"

    watermarked_text_d0_5_short = " it is also present in the public service. Public sector marketers work for government agencies, non-profit organizations, and other public institutions. Their goal is to promote the services, products, and policies of their organization to the public."
    unwatermarked_text_d0_5_short = " the reality is people with sales and marketing backgrounds are hired by government agencies in a number of capacities. Government agencies at the local, state and federal level all employ marketing professionals in areas including, but not limited to, public relations, property disposal, bond sales and purchasing."

    # standard_visualize(myWatermark, watermarked_text_d0_5_short, unwatermarked_text_d0_5_short, delta=delta)
    signature_visualize(myWatermark, watermarked_text_d0_5_short, unwatermarked_text_d0_5_short)