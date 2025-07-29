
class train_model():
    def __init__(self, num_epochs, model, train_loader, val_loader, criterion, optimizer, device):
        self.num_epochs = num_epochs
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
    def train(self, skip_eval: bool = False):
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()  # Set the model to training mode
            train_loss = 0  # Track total training loss for this epoch
            for tensors, labels in self.train_loader:
                tensors, labels = tensors.to(self.device), labels.to(self.device)  # Move to GPU if available
                
                # Forward pass
                outputs = self.model(tensors)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward() 
                self.optimizer.step()
                
                train_loss += loss.item()  # Accumulate training loss

            # Calculate average training loss for this epoch
            avg_train_loss = train_loss / len(self.train_loader)

            if skip_eval:
                continue
        
            # Validation phase
            self.model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
            val_loss = 0  # Track total validation loss for this epoch
            with self.torch.no_grad():  # Disable gradient calculation for validation
                for val_tensors, val_labels in self.val_loader:
                    val_tensors, val_labels = val_tensors.to(self.device), val_labels.to(self.device)
                    val_outputs = self.model(val_tensors)
                    val_loss += self.criterion(val_outputs, val_labels).item()  # Accumulate validation loss

            # Calculate average validation loss for this epoch
            avg_val_loss = val_loss / len(self.val_loader)

            # Print training and validation loss for the epoch
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
            
        return self.model