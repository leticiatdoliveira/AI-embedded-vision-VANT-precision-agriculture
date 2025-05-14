"""Menu system for model training options."""

# TODO: ADD A OPTION ALL TYPE OF MODELS

class MenuSystem:
    """Handles user interaction for model training options."""
    
    def __init__(self):
        """Initialize the menu system."""
        self.choices = {}
    
    def display_model_selection(self):
        """Display model selection menu and get user choice."""
        print("\n===== MODEL SELECTION MENU =====")
        print("1. Custom CNN Model")
        print("2. MobileNet Model")
        choice = input("Choose model type (1/2): ").strip()
        
        if choice == '1':
            self.choices['model_type'] = 'cnn'
            print("Selected model type: Custom CNN")
        else:
            self.choices['model_type'] = 'mobilenet'
            print("Selected model type: MobileNet")
        
        return self.choices['model_type']
    
    def display_tflite_option(self):
        """Display TFLite conversion option and get user choice."""
        print("\n===== TFLITE CONVERSION =====")
        print("Do you want to create a TFLite version of the model?")
        choice = input("Create TFLite model? (y/n): ").strip().lower()
        
        self.choices['create_tflite'] = choice == 'y' or choice == 'yes'
        print(f"Create TFLite model: {self.choices['create_tflite']}")
        
        return self.choices['create_tflite']
    
    def display_quantization_option(self):
        """Display quantization option and get user choice."""
        print("\n===== QUANTIZATION =====")
        print("Do you want to create a quantized version of the model?")
        choice = input("Create quantized model? (y/n): ").strip().lower()
        
        self.choices['create_quant'] = choice == 'y' or choice == 'yes'
        print(f"Create quantized model: {self.choices['create_quant']}")
        
        return self.choices['create_quant']
    
    def display_quant_tflite_option(self):
        """Display quantized TFLite option and get user choice."""
        if not self.choices.get('create_quant', False):
            self.choices['create_quant_tflite'] = False
            return False
            
        print("\n===== QUANTIZED TFLITE =====")
        print("Do you want to create a TFLite version of the quantized model?")
        choice = input("Create quantized TFLite model? (y/n): ").strip().lower()
        
        self.choices['create_quant_tflite'] = choice == 'y' or choice == 'yes'
        print(f"Create quantized TFLite model: {self.choices['create_quant_tflite']}")
        
        return self.choices['create_quant_tflite']
    
    def get_all_choices(self):
        """Get all menu choices at once."""
        self.display_model_selection()
        self.display_tflite_option()
        self.display_quantization_option()
        self.display_quant_tflite_option()
        
        return self.choices
